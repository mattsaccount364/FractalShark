#include "stdafx.h"

#include "Exceptions.h"
#include "GPU_Types.h"
#include "MoviePlatform.h"
#include "NvencMovieEncoder.h"
#include "ThirdParty/nvEncodeAPI.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace FractalShark {
namespace {

void
CheckCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        throw FractalSharkSeriousException(std::string("Movie CUDA: ") + cudaGetErrorString(result));
    }
}

void
CheckDriver(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        throw FractalSharkSeriousException("Movie CUDA driver error " +
                                           std::to_string(static_cast<int>(result)));
    }
}

__device__ unsigned
RoundRgb8(uint16_t sample)
{
    return (static_cast<unsigned>(sample) + 128) / 257;
}

__global__ void
ConvertMovieColors(const Color16 *colors,
                   unsigned width,
                   unsigned height,
                   unsigned char *surface,
                   size_t pitch,
                   bool rgb)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const size_t offset = y * pitch + x;
    const auto pixel = colors[static_cast<size_t>(y) * width + x];
    const float red = static_cast<float>(RoundRgb8(pixel.r));
    const float green = static_cast<float>(RoundRgb8(pixel.g));
    const float blue = static_cast<float>(RoundRgb8(pixel.b));
    const size_t planeBytes = pitch * height;
    if (rgb) {
        // The SDK's three-plane format is used with identity matrix signaling
        // to store G/B/R directly, the same mapping used by the previous path.
        surface[offset] = static_cast<unsigned char>(green);
        surface[offset + planeBytes] = static_cast<unsigned char>(blue);
        surface[offset + 2 * planeBytes] = static_cast<unsigned char>(red);
        return;
    }

    // BT.709 limited range. Average each 2x2 cell before quantizing its two
    // chroma samples. Even dimensions are checked before allocating NV12.
    constexpr float kr = 0.2126f;
    constexpr float kg = 0.7152f;
    constexpr float kb = 0.0722f;
    surface[offset] = static_cast<unsigned char>(
        16 + __float2int_rn((kr * red + kg * green + kb * blue) * (219.0f / 255)));
    if ((x & 1) == 0 && (y & 1) == 0) {
        float redSum = 0;
        float greenSum = 0;
        float blueSum = 0;
        for (unsigned row = 0; row < 2; ++row) {
            for (unsigned column = 0; column < 2; ++column) {
                const auto p = colors[static_cast<size_t>(y + row) * width + x + column];
                redSum += static_cast<float>(RoundRgb8(p.r));
                greenSum += static_cast<float>(RoundRgb8(p.g));
                blueSum += static_cast<float>(RoundRgb8(p.b));
            }
        }
        const float luma = kr * redSum + kg * greenSum + kb * blueSum;
        const size_t uv = planeBytes + (y / 2) * pitch + x;
        surface[uv] = static_cast<unsigned char>(
            128 + __float2int_rn((blueSum - luma) * (224.0f / (8 * 255 * (1 - kb)))));
        surface[uv + 1] = static_cast<unsigned char>(
            128 + __float2int_rn((redSum - luma) * (224.0f / (8 * 255 * (1 - kr)))));
    }
}

} // namespace

struct NvencMovieEncoder::Impl {
    struct Surface {
        CUdeviceptr Pixels = 0;
        size_t Pitch = 0;
        NV_ENC_REGISTERED_PTR Registered = nullptr;
        NV_ENC_OUTPUT_PTR Bitstream = nullptr;
#ifdef _WIN32
        HANDLE Event = nullptr;
        bool EventRegistered = false;
#endif
        Color16 *CpuUpload = nullptr;
        cudaStream_t UploadStream = nullptr;
    };

    Environment::MovieDriverLibrary Cuda{Environment::MovieDriver::Cuda};
    Environment::MovieDriverLibrary Nvenc{Environment::MovieDriver::Nvenc};
    decltype(&cuMemAllocPitch) AllocPitch =
        reinterpret_cast<decltype(AllocPitch)>(Cuda.Symbol("cuMemAllocPitch_v2"));
    decltype(&cuMemFree) Free = reinterpret_cast<decltype(Free)>(Cuda.Symbol("cuMemFree_v2"));
    NV_ENCODE_API_FUNCTION_LIST Api{};
    void *Session = nullptr;
    std::array<Surface, SurfaceCount> Surfaces{};
    unsigned Width = 0;
    unsigned Height = 0;
    uint32_t GopFrames = 0;
    bool Rgb = false;
    bool Async = false;
    NV_ENC_BUFFER_FORMAT Format = NV_ENC_BUFFER_FORMAT_NV12;

    ~Impl()
    {
        if (!Session) {
            return;
        }
        cudaSetDevice(0);
        for (auto &surface : Surfaces) {
#ifdef _WIN32
            if (surface.EventRegistered) {
                NV_ENC_EVENT_PARAMS eventParams{NV_ENC_EVENT_PARAMS_VER};
                eventParams.completionEvent = surface.Event;
                Api.nvEncUnregisterAsyncEvent(Session, &eventParams);
            }
            if (surface.Event) {
                CloseHandle(surface.Event);
            }
#endif
            if (surface.Registered) {
                Api.nvEncUnregisterResource(Session, surface.Registered);
            }
            if (surface.Bitstream) {
                Api.nvEncDestroyBitstreamBuffer(Session, surface.Bitstream);
            }
            if (surface.Pixels) {
                Free(surface.Pixels);
            }
            if (surface.UploadStream) {
                cudaStreamDestroy(surface.UploadStream);
            }
            if (surface.CpuUpload) {
                cudaFree(surface.CpuUpload);
            }
        }
        Api.nvEncDestroyEncoder(Session);
    }

    void
    Check(NVENCSTATUS status, const char *operation) const
    {
        if (status != NV_ENC_SUCCESS) {
            const char *detail = Session ? Api.nvEncGetLastErrorString(Session) : nullptr;
            throw FractalSharkSeriousException(std::string("NVENC ") + operation + " failed (" +
                                               std::to_string(status) + ")" +
                                               (detail ? std::string(" ") + detail : ""));
        }
    }

    int
    Cap(GUID codec, NV_ENC_CAPS cap) const
    {
        NV_ENC_CAPS_PARAM query{NV_ENC_CAPS_PARAM_VER};
        query.capsToQuery = cap;
        int value = 0;
        Check(Api.nvEncGetEncodeCaps(Session, codec, &query, &value), "capability query");
        return value;
    }

    void
    Initialize(unsigned width, unsigned height, const MovieVideoSettings &settings)
    {
        Width = width;
        Height = height;
        GopFrames = settings.GopFrames;
        Rgb = settings.Codec == MovieCodec::HevcRgbLossless;
        Format = Rgb ? NV_ENC_BUFFER_FORMAT_YUV444 : NV_ENC_BUFFER_FORMAT_NV12;
        if (width == 0 || height == 0 || settings.Preset < 1 || settings.Preset > 7 ||
            settings.Fps == 0 || GopFrames == 0 || (!Rgb && ((width | height) & 1)) ||
            settings.Codec == MovieCodec::PngOnly) {
            throw FractalSharkSeriousException(
                "Invalid NVENC configuration (NV12 needs even dimensions)");
        }
        CheckCuda(cudaSetDevice(0));
        CheckCuda(cudaFree(nullptr));
        using ContextQuery = CUresult(CUDAAPI *)(CUcontext *);
        const auto getContext = reinterpret_cast<ContextQuery>(Cuda.Symbol("cuCtxGetCurrent"));
        CUcontext context = nullptr;
        CheckDriver(getContext(&context));
        if (!context) {
            throw FractalSharkSeriousException("NVENC requires the CUDA render device context");
        }

        const auto maxVersion = reinterpret_cast<decltype(&NvEncodeAPIGetMaxSupportedVersion)>(
            Nvenc.Symbol("NvEncodeAPIGetMaxSupportedVersion"));
        uint32_t version = 0;
        Check(maxVersion(&version), "driver version query");
        const uint32_t required = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
        if (version < required) {
            throw FractalSharkSeriousException("The NVIDIA driver must support NVENC API 13.1");
        }
        const auto createApi = reinterpret_cast<decltype(&NvEncodeAPICreateInstance)>(
            Nvenc.Symbol("NvEncodeAPICreateInstance"));
        Api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        Check(createApi(&Api), "function table creation");
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open{NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
        open.device = context;
        open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        open.apiVersion = NVENCAPI_VERSION;
        Check(Api.nvEncOpenEncodeSessionEx(&open, &Session), "CUDA session creation");

        const bool h264 = settings.Codec == MovieCodec::H264;
        const GUID codec = h264 ? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
        if (width > static_cast<unsigned>(Cap(codec, NV_ENC_CAPS_WIDTH_MAX)) ||
            height > static_cast<unsigned>(Cap(codec, NV_ENC_CAPS_HEIGHT_MAX)) ||
            (Rgb && (!Cap(codec, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ||
                     !Cap(codec, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)))) {
            throw FractalSharkSeriousException(
                "This NVIDIA device cannot encode the selected movie profile at the requested size");
        }

        const std::array presets{NV_ENC_PRESET_P1_GUID,
                                 NV_ENC_PRESET_P2_GUID,
                                 NV_ENC_PRESET_P3_GUID,
                                 NV_ENC_PRESET_P4_GUID,
                                 NV_ENC_PRESET_P5_GUID,
                                 NV_ENC_PRESET_P6_GUID,
                                 NV_ENC_PRESET_P7_GUID};
        const auto preset = presets[settings.Preset - 1];
        const auto tune = Rgb ? NV_ENC_TUNING_INFO_LOSSLESS : NV_ENC_TUNING_INFO_HIGH_QUALITY;
        NV_ENC_PRESET_CONFIG presetConfig{NV_ENC_PRESET_CONFIG_VER};
        presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
        Check(Api.nvEncGetEncodePresetConfigEx(Session, codec, preset, tune, &presetConfig),
              "preset selection");

        auto &config = presetConfig.presetCfg;
        config.profileGUID = h264  ? NV_ENC_H264_PROFILE_HIGH_GUID
                             : Rgb ? NV_ENC_HEVC_PROFILE_FREXT_GUID
                                   : NV_ENC_HEVC_PROFILE_MAIN_GUID;
        // Application-scheduled IDRs preserve their absolute cadence after a
        // resume. P-only output associates each submitted frame with one AU.
        config.gopLength = NVENC_INFINITE_GOPLENGTH;
        config.frameIntervalP = 1;
        auto &rc = config.rcParams;
        rc.enableLookahead = 0;
        rc.disableIadapt = 1;
        rc.zeroReorderDelay = 1;
        rc.rateControlMode = h264 ? NV_ENC_PARAMS_RC_VBR : NV_ENC_PARAMS_RC_CONSTQP;
        rc.multiPass = NV_ENC_MULTI_PASS_DISABLED;
        if (h264) {
            rc.targetQuality = static_cast<uint8_t>(settings.Quality);
            rc.averageBitRate = 0;
            rc.maxBitRate = 0;
        } else {
            const auto qp = Rgb ? 0u : settings.Quality;
            rc.constQP = {qp, qp, qp};
            rc.enableAQ = 0;
            rc.enableTemporalAQ = 0;
        }
        NV_ENC_CONFIG_H264_VUI_PARAMETERS *vui = nullptr;
        if (h264) {
            auto &h = config.encodeCodecConfig.h264Config;
            h.idrPeriod = NVENC_INFINITE_GOPLENGTH;
            h.outputAUD = 1;
            h.repeatSPSPPS = 1;
            h.chromaFormatIDC = 1;
            h.inputBitDepth = h.outputBitDepth = NV_ENC_BIT_DEPTH_8;
            vui = &h.h264VUIParameters;
        } else {
            auto &h = config.encodeCodecConfig.hevcConfig;
            h.idrPeriod = NVENC_INFINITE_GOPLENGTH;
            h.outputAUD = 1;
            h.repeatSPSPPS = 1;
            h.chromaFormatIDC = Rgb ? 3 : 1;
            h.inputBitDepth = h.outputBitDepth = NV_ENC_BIT_DEPTH_8;
            vui = &h.hevcVUIParameters;
        }
        vui->videoSignalTypePresentFlag = 1;
        vui->videoFullRangeFlag = Rgb;
        vui->colourDescriptionPresentFlag = 1;
        vui->colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        vui->transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        vui->colourMatrix = Rgb ? NV_ENC_VUI_MATRIX_COEFFS_RGB : NV_ENC_VUI_MATRIX_COEFFS_BT709;

#ifdef _WIN32
        Async = Cap(codec, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT) != 0;
#endif
        NV_ENC_INITIALIZE_PARAMS init{NV_ENC_INITIALIZE_PARAMS_VER};
        init.encodeGUID = codec;
        init.presetGUID = preset;
        init.tuningInfo = tune;
        init.encodeWidth = width;
        init.encodeHeight = height;
        init.darWidth = width;
        init.darHeight = height;
        init.frameRateNum = settings.Fps;
        init.frameRateDen = 1;
        init.enableEncodeAsync = Async;
        init.enablePTD = 1;
        init.encodeConfig = &config;
        Check(Api.nvEncInitializeEncoder(Session, &init), "initialization");

        for (auto &surface : Surfaces) {
            const size_t rows = Rgb ? 3ull * height : height + height / 2;
            CheckDriver(AllocPitch(&surface.Pixels, &surface.Pitch, width, rows, 16));
            NV_ENC_REGISTER_RESOURCE resource{NV_ENC_REGISTER_RESOURCE_VER};
            resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
            resource.resourceToRegister = reinterpret_cast<void *>(surface.Pixels);
            resource.width = width;
            resource.height = height;
            resource.pitch = static_cast<uint32_t>(surface.Pitch);
            resource.bufferFormat = Format;
            resource.bufferUsage = NV_ENC_INPUT_IMAGE;
            Check(Api.nvEncRegisterResource(Session, &resource), "surface registration");
            surface.Registered = resource.registeredResource;

            NV_ENC_CREATE_BITSTREAM_BUFFER output{NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
            Check(Api.nvEncCreateBitstreamBuffer(Session, &output), "bitstream allocation");
            surface.Bitstream = output.bitstreamBuffer;
#ifdef _WIN32
            if (Async) {
                surface.Event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                if (!surface.Event) {
                    throw FractalSharkSeriousException("Cannot create NVENC completion event");
                }
                NV_ENC_EVENT_PARAMS eventParams{NV_ENC_EVENT_PARAMS_VER};
                eventParams.completionEvent = surface.Event;
                Check(Api.nvEncRegisterAsyncEvent(Session, &eventParams), "event registration");
                surface.EventRegistered = true;
            }
#endif
        }
    }
};

NvencMovieEncoder::NvencMovieEncoder(unsigned width, unsigned height, const MovieVideoSettings &settings)
    : m_Impl(std::make_unique<Impl>())
{
    m_Impl->Initialize(width, height, settings);
}

NvencMovieEncoder::~NvencMovieEncoder() = default;

uint32_t
NvencMovieEncoder::WriteGpu(size_t slot, const Color16 *colors, void *streamHandle)
{
    const auto stream = static_cast<cudaStream_t>(streamHandle);
    const auto &surface = m_Impl->Surfaces.at(slot);
    constexpr dim3 block(16, 16);
    const dim3 grid((m_Impl->Width + block.x - 1) / block.x, (m_Impl->Height + block.y - 1) / block.y);
    ConvertMovieColors<<<grid, block, 0, stream>>>(colors,
                                                   m_Impl->Width,
                                                   m_Impl->Height,
                                                   reinterpret_cast<unsigned char *>(surface.Pixels),
                                                   surface.Pitch,
                                                   m_Impl->Rgb);
    const auto error = cudaGetLastError();
    // Fence only the producing renderer before the ownership handoff. This
    // also lets the renderer reuse its colors while NVENC reads the surface.
    const auto fence = cudaStreamSynchronize(stream);
    return error == cudaSuccess ? fence : error;
}

uint32_t
NvencMovieEncoder::WriteCpu(size_t slot, std::span<const Color16> colors)
{
    auto &surface = m_Impl->Surfaces.at(slot);
    CheckCuda(cudaSetDevice(0));
    if (!surface.CpuUpload) {
        CheckCuda(cudaStreamCreateWithFlags(&surface.UploadStream, cudaStreamNonBlocking));
        CheckCuda(cudaMalloc(&surface.CpuUpload, colors.size_bytes()));
    }
    CheckCuda(cudaMemcpyAsync(surface.CpuUpload,
                              colors.data(),
                              colors.size_bytes(),
                              cudaMemcpyHostToDevice,
                              surface.UploadStream));
    return WriteGpu(slot, surface.CpuUpload, surface.UploadStream);
}

std::vector<unsigned char>
NvencMovieEncoder::Encode(size_t slot, uint64_t frame, bool keyframe)
{
    CheckCuda(cudaSetDevice(0));
    const auto &surface = m_Impl->Surfaces.at(slot);
    auto &api = m_Impl->Api;
    auto *session = m_Impl->Session;
    NV_ENC_MAP_INPUT_RESOURCE mapped{NV_ENC_MAP_INPUT_RESOURCE_VER};
    mapped.registeredResource = surface.Registered;
    m_Impl->Check(api.nvEncMapInputResource(session, &mapped), "input mapping");
    struct UnmapInput {
        NV_ENCODE_API_FUNCTION_LIST &Api;
        void *Session;
        NV_ENC_INPUT_PTR Input;
        ~UnmapInput() { Api.nvEncUnmapInputResource(Session, Input); }
    } unmap{api, session, mapped.mappedResource};

    NV_ENC_PIC_PARAMS picture{NV_ENC_PIC_PARAMS_VER};
    picture.inputWidth = m_Impl->Width;
    picture.inputHeight = m_Impl->Height;
    picture.inputPitch = static_cast<uint32_t>(surface.Pitch);
    picture.inputBuffer = mapped.mappedResource;
    picture.bufferFmt = m_Impl->Format;
    picture.outputBitstream = surface.Bitstream;
    picture.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picture.inputTimeStamp = frame;
    picture.inputDuration = 1;
    if (keyframe || frame % m_Impl->GopFrames == 0) {
        picture.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
    }
#ifdef _WIN32
    picture.completionEvent = surface.Event;
#endif
    m_Impl->Check(api.nvEncEncodePicture(session, &picture), "encode");
#ifdef _WIN32
    if (surface.Event && WaitForSingleObject(surface.Event, INFINITE) != WAIT_OBJECT_0) {
        throw FractalSharkSeriousException("Cannot wait for NVENC completion");
    }
#endif
    NV_ENC_LOCK_BITSTREAM output{NV_ENC_LOCK_BITSTREAM_VER};
    output.outputBitstream = surface.Bitstream;
    output.doNotWait = m_Impl->Async;
    m_Impl->Check(api.nvEncLockBitstream(session, &output), "bitstream read");
    struct UnlockOutput {
        NV_ENCODE_API_FUNCTION_LIST &Api;
        void *Session;
        NV_ENC_OUTPUT_PTR Output;
        ~UnlockOutput() { Api.nvEncUnlockBitstream(Session, Output); }
    } unlock{api, session, surface.Bitstream};
    if (output.bitstreamSizeInBytes == 0 || output.outputTimeStamp != frame) {
        throw FractalSharkSeriousException("NVENC returned an empty or out-of-order access unit");
    }
    const auto *bytes = static_cast<const unsigned char *>(output.bitstreamBufferPtr);
    return {bytes, bytes + output.bitstreamSizeInBytes};
}

} // namespace FractalShark
