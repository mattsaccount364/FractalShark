#pragma once

#include <array>
#include <string_view>

namespace RenderTests {
struct GoldenChecksum {
    const char *CaseId;
    const char *Crc;
};

// Shared Windows Release baseline: CRC-64 of decoded RGBA16 PNG bytes or canonical reference text.
inline constexpr std::array<GoldenChecksum, 2485> GoldenChecksums{{
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits32_Store0_Ref2_AA2_Step1_Comp20_LA0_Threads1_Load0",
     "d063f8c0550a929f"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits32_Store0_Ref2_AA3_Step1_Comp20_LA0_Threads1_Load0",
     "63bd998580612e8d"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits32_Store0_Ref2_AA4_Step1_Comp20_LA0_Threads1_Load0",
     "59c6b3cb450564c9"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits64_Store0_Ref2_AA2_Step1_Comp20_LA0_Threads1_Load0",
     "d063f8c0550a929f"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits64_Store0_Ref2_AA3_Step1_Comp20_LA0_Threads1_Load0",
     "63bd998580612e8d"},
    {"RenderGolden_Antialiasing_Gpu1x32_View0_Bits64_Store0_Ref2_AA4_Step1_Comp20_LA0_Threads1_Load0",
     "59c6b3cb450564c9"},
    {"RenderGolden_AutoSelection_AutoSelect_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8eee9f298276208d"},
    {"RenderGolden_AutoSelection_AutoSelect_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_AutoSelection_AutoSelect_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8eee9f298276208d"},
    {"RenderGolden_AutoSelection_AutoSelect_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_AutoSelection_AutoSelect_View1_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "046ccafe17946e84"},
    {"RenderGolden_AutoSelection_AutoSelect_View1_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "bd4a1dbf5054bcff"},
    {"RenderGolden_AutoSelection_AutoSelect_View1_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "046ccafe17946e84"},
    {"RenderGolden_AutoSelection_AutoSelect_View1_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "bd4a1dbf5054bcff"},
    {"RenderGolden_AutoSelection_AutoSelect_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_AutoSelection_AutoSelect_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_AutoSelection_AutoSelect_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_AutoSelection_AutoSelect_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_AutoSelection_AutoSelect_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "53c17b6ed0107da9"},
    {"RenderGolden_AutoSelection_AutoSelect_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "c60ffd705a8aca42"},
    {"RenderGolden_AutoSelection_AutoSelect_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "53c17b6ed0107da9"},
    {"RenderGolden_AutoSelection_AutoSelect_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_AutoGpu",
     "c60ffd705a8aca42"},
    {"RenderGolden_Basic_AutoSelect_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_AutoSelect_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_AutoSelect_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_AutoSelect_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_AutoSelect_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "INCOMPLETE"},
    {"RenderGolden_Basic_AutoSelect_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "INCOMPLETE"},
    {"RenderGolden_Basic_AutoSelect_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "INCOMPLETE"},
    {"RenderGolden_Basic_AutoSelect_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "INCOMPLETE"},
    {"RenderGolden_Basic_AutoSelect_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_AutoSelect_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_AutoSelect_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_AutoSelect_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "AutoGpu",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAHDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAHDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAHDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAHDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "00579b31ca3aa41e"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "00579b31ca3aa41e"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "00579b31ca3aa41e"},
    {"RenderGolden_Basic_Cpu32PerturbedBLAV2HDR_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "00579b31ca3aa41e"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "43fdb635f9fe669c"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3a8313703a7428b6"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3a8313703a7428b6"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3a8313703a7428b6"},
    {"RenderGolden_Basic_Cpu32PerturbedRCBLAV2HDR_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3a8313703a7428b6"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAHDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAHDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAHDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAHDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_Basic_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "88184a0787aa341c"},
    {"RenderGolden_Basic_Cpu64PerturbedBLA_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLA_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLA_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedBLA_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3619b44004dd609b"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2258571d542f1530"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2258571d542f1530"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2258571d542f1530"},
    {"RenderGolden_Basic_Cpu64PerturbedRCBLAV2HDR_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2258571d542f1530"},
    {"RenderGolden_Basic_Cpu64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_Cpu64_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_Cpu64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_Cpu64_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_CpuHDR32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "accf6b90d46ad55e"},
    {"RenderGolden_Basic_CpuHDR32_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "accf6b90d46ad55e"},
    {"RenderGolden_Basic_CpuHDR32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "accf6b90d46ad55e"},
    {"RenderGolden_Basic_CpuHDR32_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "accf6b90d46ad55e"},
    {"RenderGolden_Basic_CpuHDR64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_CpuHDR64_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_CpuHDR64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_CpuHDR64_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8eee9f298276208d"},
    {"RenderGolden_Basic_CpuHigh_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "37bb2fcca025367b"},
    {"RenderGolden_Basic_CpuHigh_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "37bb2fcca025367b"},
    {"RenderGolden_Basic_CpuHigh_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "37bb2fcca025367b"},
    {"RenderGolden_Basic_CpuHigh_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "37bb2fcca025367b"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "57820dd8dc0a28e7"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "57820dd8dc0a28e7"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "57820dd8dc0a28e7"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2LAO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "57820dd8dc0a28e7"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "c7b8185405a684cb"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "c7b8185405a684cb"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "c7b8185405a684cb"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2PO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "c7b8185405a684cb"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "03040294261d819a"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "03040294261d819a"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "03040294261d819a"},
    {"RenderGolden_Basic_Gpu1x32PerturbedLAv2_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "03040294261d819a"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2LAO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "2cc7dc80afe25426"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "2cc7dc80afe25426"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "2cc7dc80afe25426"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2PO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "2cc7dc80afe25426"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2a554db9f2e6c6dd"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "6d4d2e8fcd90816f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "6d4d2e8fcd90816f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "6d4d2e8fcd90816f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedRCLAv2_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "6d4d2e8fcd90816f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "fca8dd146910aa44"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "fca8dd146910aa44"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "fca8dd146910aa44"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "fca8dd146910aa44"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "f08c5e53697db44f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "f08c5e53697db44f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "f08c5e53697db44f"},
    {"RenderGolden_Basic_Gpu1x32PerturbedScaled_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "f08c5e53697db44f"},
    {"RenderGolden_Basic_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_Gpu1x32_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_Gpu1x32_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "5d387238abb77f8b"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "5d387238abb77f8b"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "5d387238abb77f8b"},
    {"RenderGolden_Basic_Gpu1x64PerturbedBLA_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "5d387238abb77f8b"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "aaa64f66346466c5"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "aaa64f66346466c5"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "aaa64f66346466c5"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2LAO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "aaa64f66346466c5"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "e836bfa6fcaaabe9"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "e836bfa6fcaaabe9"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "e836bfa6fcaaabe9"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "e836bfa6fcaaabe9"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "c070612cfacbdc58"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "c070612cfacbdc58"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "c070612cfacbdc58"},
    {"RenderGolden_Basic_Gpu1x64PerturbedLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "c070612cfacbdc58"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9a6a74feaeeedfeb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9a6a74feaeeedfeb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9a6a74feaeeedfeb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2LAO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9a6a74feaeeedfeb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "74eb33a4c3c03bc8"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "74eb33a4c3c03bc8"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "74eb33a4c3c03bc8"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "74eb33a4c3c03bc8"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "2547a04ce83b3c97"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1fa413faf41d1adb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1fa413faf41d1adb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1fa413faf41d1adb"},
    {"RenderGolden_Basic_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1fa413faf41d1adb"},
    {"RenderGolden_Basic_Gpu1x64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Basic_Gpu1x64_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Basic_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Basic_Gpu1x64_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "467e196cf7246532"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "467e196cf7246532"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "467e196cf7246532"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2LAO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "467e196cf7246532"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "3756774c527ea068"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "3756774c527ea068"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "3756774c527ea068"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2PO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "3756774c527ea068"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "7b4e3951576a9dd1"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "7b4e3951576a9dd1"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "7b4e3951576a9dd1"},
    {"RenderGolden_Basic_Gpu2x32PerturbedLAv2_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "7b4e3951576a9dd1"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2LAO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "9af09b1c8da12e1f"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "32d6e6f1e9c7a8dc"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "32d6e6f1e9c7a8dc"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "32d6e6f1e9c7a8dc"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2PO_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Iters50000",
     "32d6e6f1e9c7a8dc"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "edbdf54e3a370ec6"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View9_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "ac2e2cdd5fce9d7e"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View9_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "ac2e2cdd5fce9d7e"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View9_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "ac2e2cdd5fce9d7e"},
    {"RenderGolden_Basic_Gpu2x32PerturbedRCLAv2_View9_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0_Iters50000",
     "ac2e2cdd5fce9d7e"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32PerturbedScaled_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_Gpu2x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Basic_Gpu2x32_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Basic_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Basic_Gpu2x32_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Basic_Gpu2x64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "32f04db5085f92bb"},
    {"RenderGolden_Basic_Gpu2x64_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "32f04db5085f92bb"},
    {"RenderGolden_Basic_Gpu2x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "32f04db5085f92bb"},
    {"RenderGolden_Basic_Gpu2x64_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "32f04db5085f92bb"},
    {"RenderGolden_Basic_Gpu4x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8cc57d8b06972ddd"},
    {"RenderGolden_Basic_Gpu4x32_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8cc57d8b06972ddd"},
    {"RenderGolden_Basic_Gpu4x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8cc57d8b06972ddd"},
    {"RenderGolden_Basic_Gpu4x32_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "8cc57d8b06972ddd"},
    {"RenderGolden_Basic_Gpu4x64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "73e1653adf0c373e"},
    {"RenderGolden_Basic_Gpu4x64_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "73e1653adf0c373e"},
    {"RenderGolden_Basic_Gpu4x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "73e1653adf0c373e"},
    {"RenderGolden_Basic_Gpu4x64_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "73e1653adf0c373e"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6c4962034601d9df"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6c4962034601d9df"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6c4962034601d9df"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6c4962034601d9df"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6b1cea3b7a895943"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6b1cea3b7a895943"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6b1cea3b7a895943"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "6b1cea3b7a895943"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "54607c2fe7bca6f6"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "54607c2fe7bca6f6"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "54607c2fe7bca6f6"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "54607c2fe7bca6f6"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e830035f35408f4e"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e830035f35408f4e"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e830035f35408f4e"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e830035f35408f4e"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3653e68932cd08e9"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3653e68932cd08e9"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3653e68932cd08e9"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3653e68932cd08e9"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "cbeb5d705c443794"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "cbeb5d705c443794"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "cbeb5d705c443794"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "cbeb5d705c443794"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View27_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View27_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9ee5687d10eea487"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9ee5687d10eea487"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9ee5687d10eea487"},
    {"RenderGolden_Basic_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "9ee5687d10eea487"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "46a346c2acdd72d7"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "46a346c2acdd72d7"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "46a346c2acdd72d7"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "46a346c2acdd72d7"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1cf0625d7158a2bb"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1cf0625d7158a2bb"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1cf0625d7158a2bb"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedBLA_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "1cf0625d7158a2bb"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5a1a68c9f628533b"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5a1a68c9f628533b"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5a1a68c9f628533b"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5a1a68c9f628533b"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8246d961ae550d64"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8246d961ae550d64"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8246d961ae550d64"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8246d961ae550d64"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8530650d919679d8"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8530650d919679d8"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8530650d919679d8"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "8530650d919679d8"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "b15380b9fe208644"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb5c0129d9d35b90"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb5c0129d9d35b90"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb5c0129d9d35b90"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb5c0129d9d35b90"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5267d50602a85cc6"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5267d50602a85cc6"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5267d50602a85cc6"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5267d50602a85cc6"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b07c2fa7bea0c162"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e6b4d8292b0e47fc"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e6b4d8292b0e47fc"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e6b4d8292b0e47fc"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "e6b4d8292b0e47fc"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5364e31b5b74c71d"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5364e31b5b74c71d"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb26bab1e662bb63"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "bb26bab1e662bb63"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "c0cfabe16fcef282"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "c0cfabe16fcef282"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "c0cfabe16fcef282"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "c0cfabe16fcef282"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "05742cf48413662f"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "05742cf48413662f"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "05742cf48413662f"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "05742cf48413662f"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "fddeb815d38e3e35"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "fddeb815d38e3e35"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "fddeb815d38e3e35"},
    {"RenderGolden_Basic_GpuHDRx32PerturbedScaled_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "fddeb815d38e3e35"},
    {"RenderGolden_Basic_GpuHDRx32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Basic_GpuHDRx32_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Basic_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Basic_GpuHDRx32_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "65f0a66a25ef28cc"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "65f0a66a25ef28cc"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "65f0a66a25ef28cc"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "65f0a66a25ef28cc"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3530de8bd1d666aa"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3530de8bd1d666aa"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3530de8bd1d666aa"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedBLA_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "3530de8bd1d666aa"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "5de4ff96ca9467be"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "53424444659ba7e0"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "53424444659ba7e0"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "53424444659ba7e0"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "53424444659ba7e0"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3b00d515fe28f70b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3b00d515fe28f70b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3b00d515fe28f70b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "3b00d515fe28f70b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "0bdde180bc6c013e"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "0bdde180bc6c013e"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "0bdde180bc6c013e"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "0bdde180bc6c013e"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "991bea00f2bc22fd"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "991bea00f2bc22fd"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "991bea00f2bc22fd"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_Threads1_"
     "Load0",
     "991bea00f2bc22fd"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "554343e3db1ae089"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2LAO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "076871aa0e17d8ba"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "dcd9d5f20a259d0c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "dcd9d5f20a259d0c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "dcd9d5f20a259d0c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "dcd9d5f20a259d0c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "1ac929aec82f2435"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "1ac929aec82f2435"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "1ac929aec82f2435"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2PO_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "1ac929aec82f2435"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View0_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "133514f8b492253c"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View10_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View10_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View10_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View11_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b3d889e50bd4d66b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View11_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b3d889e50bd4d66b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View11_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b3d889e50bd4d66b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View11_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "b3d889e50bd4d66b"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "34332c958151daae"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View5_Bits32_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "34332c958151daae"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "34332c958151daae"},
    {"RenderGolden_Basic_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store2_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "34332c958151daae"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "b64150409176794a"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "be1bee334185ad79"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "0e09d01ce50bf24d"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "7686be236a566194"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "a422b2f8435facc8"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "a4d49b84121033cc"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "0501567ebefe3c56"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "8a45ecc3dc2d26ea"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "332cc1e873db7b75"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "290dae6875ac9e53"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "c20d96f4f6e3ec7b"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "f76cb61610eb1f8d"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "c20d96f4f6e3ec7b"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "c20d96f4f6e3ec7b"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "7d05ed0859888bec"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "f18c7d33cf1602aa"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "516da9791046bc60"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "3e517901b2d50105"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "957f49231eb206a9"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "2939270a89371dcc"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_Compression_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "cf8d7f3e8d8af2f8"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "9e88c88f7ac46f5e"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "50e9bedb990832ea"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "883d62311978a9e5"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "0afd40557d646cb8"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "dd33adf0607cb066"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "f298fd96a6ed5a73"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "d38330c47db2a868"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "e65e329590e57404"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "54026d34471b3258"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "fa49bc6df61b62c8"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "2b72269cf3efeac0"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "aa9df4f02c4f0bda"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "3c32cab2300b57ac"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "16b1d074f7cb534f"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "4117c9697e84cc9a"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "9d60e548adc3de60"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "3bb6a547f326ff7c"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "7bc0e440de289637"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "e9b0a81cfe1f41ed"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "5e864a16c1a6a151"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "086e6df8d82dac96"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "a1af5bda205c87ec"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "1fa413faf41d1adb"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "a2daab257a3ec1d9"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "5726adf3463ca4f2"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "f5765a1ca84364d8"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "26e5b02768679e94"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "10ef917c694b43ce"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "63f47e942444c204"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "7fa34dd1638e76d1"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "340df171a93b0eb6"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "a1249e1d0443f9b3"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "38cc06210465ae65"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "b390fa00dc403696"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "90ce2877e7dbbd0d"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "45b130ad88f82ca4"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "5937e515e6c7f102"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "3ddd0b78b270126a"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "58a9a405e761141f"},
    {"RenderGolden_Compression_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "5f0b0420a365fe5d"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "1533de080a843a77"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "7aabc8671c700c4e"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "7aabc8671c700c4e"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "7aabc8671c700c4e"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "1acf9e61df5ef954"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "bcf898c6a9ff7a14"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "e5dbfa35eec8ab8c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "7ed78700db2345ac"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "5032b6722f196dd1"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "c2310b67ffbb4613"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "81e6d6004ab25a85"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "c2310b67ffbb4613"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "c2310b67ffbb4613"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "23adaf3218ee5efd"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "23adaf3218ee5efd"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "23adaf3218ee5efd"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "412e2b3e00b1acd2"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "0bbdc09e61247172"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "0d8f47b82f062619"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "54321aa7098cef23"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "aa062b14b037de7c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "639430974d104435"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "c5434be0bc681ee8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "9623b29d77808b78"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "b01f2f9893afde80"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "37ac44e91d665bf7"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "e3aa1f781df53ec5"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "ce9b3a0b1667cde3"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "fb1f813002462887"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "9e9eab56b3cc9534"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "44c37585fb66796f"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "61a3491c82c80ade"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "ec347718491fe3bb"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "e00ad8cde4d817be"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "0057a25689983762"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "c8360b430acd3f83"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "f92223b34eb7defb"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "bc17ac08c67cf0cf"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "c6f0e2bde026764a"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "93f4e599e04e83e4"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "ff9b41d7f39a9e1b"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "d22556a0dfa18151"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "9ee5687d10eea487"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "e0cbf2c8ca03b31c"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "3eddfa9a6e238aed"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "2e1b6b0e6d7c3f82"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "65b028a757c802ba"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "957be35d97db057f"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "9b5d7b70ae0262c0"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "e8042bd48a941a4a"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "d83b07732a89e181"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "1e2af0ac77050de0"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "f16d668086d9b34b"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "62e625eb0f0c4db5"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "12e1d0d12cbceade"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "a2d058deccdf967f"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "ca7c2d22c9baf315"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "e18a74443563c8b0"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "f7e9514b4568a9cf"},
    {"RenderGolden_Compression_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "eff381da06dfd442"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "b61d11986f808e10"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "d3d66172d6aa6aad"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "b8de6b0246add2ab"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "5935bd4126fb9df8"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "283814f08aed7803"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "543e8b7a92cebb26"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "b5f6ed1e002b2d07"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "199f0e66b9cc8997"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "199f0e66b9cc8997"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "b7a5f43534730536"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "199f0e66b9cc8997"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "44156a917bcea4ce"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "4b72fe4239930575"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "fd95a87fa7d25143"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "0df52b21b2beadc1"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "7c8a3672b4ce604a"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "e041db217cc31875"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "f6deb24c30ce2382"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "c53a375c901fa794"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "54adb3922567af09"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "aa06475b8ba95d7c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "9ffc255301236bc9"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "def708f00072b11f"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "d23358ce8a0c9d36"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "01400237edc24183"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "0d1b3185266427ce"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "e2b88b062d6aa9c9"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "07ab21cac27331b5"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "82df0492954ae941"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "a42e3993f97d663d"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "0749765c16cac0ce"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "07becf05571792af"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "97452483f9a4ef47"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "0413a11ce774ca38"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "94f436b50b023837"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "b0c8dc1ebe0e749f"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "bb26bab1e662bb63"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "b0c8dc1ebe0e749f"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "bb26bab1e662bb63"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "71844202fc0bdc13"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "6777a700f8e9dcd3"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "b0c8dc1ebe0e749f"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "bb26bab1e662bb63"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "0bfe0c23e799de07"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "d79292036184c74e"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "8df07a86fd9f55a0"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "8c24d4be492d8752"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "ddc8f55a5ab702f2"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "610d84453a1f0a57"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "209cac0455b11615"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "ac1a8b82158d7f69"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "8aeaa5ac3c7d8bce"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "b4b99875c4957798"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "291d37d92e400a69"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "230bbc7e30370e5f"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "1c51faa4a62d2ad3"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "5c751e21e2237307"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "d246e27dd7bde730"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "3a2398654e02c269"},
    {"RenderGolden_Compression_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "946f9804d772ac92"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "b7afa91385a9ad9a"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "f16dab136b1014e2"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "f16dab136b1014e2"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "f16dab136b1014e2"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "efd2121674c7b2a8"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "164bf2c3efcfc25a"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "ba8a0002a79f26f6"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "47d97d519b578fca"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "62f88686ec72da01"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "5892892b0605bf45"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "de569a2881ec0dec"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "5892892b0605bf45"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "5892892b0605bf45"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "9bb33e3f4cc940bc"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "9bb33e3f4cc940bc"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "9bb33e3f4cc940bc"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "39b49be89fe5a11a"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "52ca41260ca92cbb"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "10f349c2a4d89367"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Decompressed",
     "5ebc1e9b080cd542"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Original",
     "8050c742b7dfd048"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp10_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Decompressed",
     "718c98c24948693d"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Original",
     "1a9e7dcf96d61794"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp11_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Decompressed",
     "6e6dceca00237c40"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Original",
     "4af46c8aa332b6cb"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp12_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Decompressed",
     "ff6f59e8ed0e69a1"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Original",
     "8746677e03981697"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp13_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Decompressed",
     "07be65858ceeb61e"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Original",
     "85ced31ec44cd54d"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp14_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Decompressed",
     "65315e56714b0793"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Original",
     "cce61ce28d6a5e70"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp15_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Decompressed",
     "b42516b945d91e90"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Original",
     "1f20a2fce54c4217"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp16_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Decompressed",
     "6abb6ffeaf278f9d"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Original",
     "838eae160675ae0b"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp17_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Decompressed",
     "01beb59666df1001"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Original",
     "5a82fdd89b116b4d"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Decompressed",
     "9ac81c97be4cc179"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Original",
     "02b6ef261812e5bc"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Decompressed",
     "c922f6b0b0f6ca0b"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Original",
     "af5364d724761014"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp1_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "389a50fd8c20f3c4"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "34332c958151daae"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Decompressed",
     "3c585ddda4d0a4d5"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Original",
     "80b7404974aaa0d2"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp2_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Decompressed",
     "2a6f040630da5d5f"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Original",
     "d6bd95caf9c4b6bc"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp3_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Decompressed",
     "62332ca5cdaa03db"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Original",
     "94fb1928f602bac2"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp4_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Decompressed",
     "fff3782e3a043926"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Original",
     "6cf1a04c4c97aee0"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp5_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Decompressed",
     "2cbd1210b2ab8b30"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Original",
     "9c007ede584aa24e"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp6_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Decompressed",
     "daf081dcdadf5a33"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Original",
     "9e664d56515077a7"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp7_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Decompressed",
     "e0b7e08631e48843"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Original",
     "0852c00db25984b7"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp8_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Decompressed",
     "fb8ee6660650e074"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Original",
     "9af8b2e8d68a14f0"},
    {"RenderGolden_Compression_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp9_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "a1733492a246bb8e"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "88184a0787aa341c"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "a1733492a246bb8e"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "88184a0787aa341c"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "a1733492a246bb8e"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "88184a0787aa341c"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "172a5b3ddc6a22c5"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "88184a0787aa341c"},
    {"RenderGolden_CpuReferenceSave_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "3619b44004dd609b"},
    {"RenderGolden_GpuColors_AA1_Bits32_Pass0", "b429ab2a854a6f9b"},
    {"RenderGolden_GpuColors_AA1_Bits32_Pass1", "b429ab2a854a6f9b"},
    {"RenderGolden_GpuColors_AA1_Bits64_Pass0", "b429ab2a854a6f9b"},
    {"RenderGolden_GpuColors_AA1_Bits64_Pass1", "b429ab2a854a6f9b"},
    {"RenderGolden_GpuColors_AA2_Bits32_Pass0", "de132ba6cb67b61f"},
    {"RenderGolden_GpuColors_AA2_Bits32_Pass1", "de132ba6cb67b61f"},
    {"RenderGolden_GpuColors_AA2_Bits64_Pass0", "de132ba6cb67b61f"},
    {"RenderGolden_GpuColors_AA2_Bits64_Pass1", "de132ba6cb67b61f"},
    {"RenderGolden_GpuColors_AA3_Bits32_Pass0", "6410c065e1ed8d70"},
    {"RenderGolden_GpuColors_AA3_Bits32_Pass1", "6410c065e1ed8d70"},
    {"RenderGolden_GpuColors_AA3_Bits64_Pass0", "6410c065e1ed8d70"},
    {"RenderGolden_GpuColors_AA3_Bits64_Pass1", "6410c065e1ed8d70"},
    {"RenderGolden_GpuColors_AA4_Bits32_Pass0", "6e87284e076e38a2"},
    {"RenderGolden_GpuColors_AA4_Bits32_Pass1", "6e87284e076e38a2"},
    {"RenderGolden_GpuColors_AA4_Bits64_Pass0", "6e87284e076e38a2"},
    {"RenderGolden_GpuColors_AA4_Bits64_Pass1", "6e87284e076e38a2"},
    {"RenderGolden_GpuReferenceBucket_1024_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_1024_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_1024_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_131072_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_131072_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_131072_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_16384_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_16384_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_16384_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_2048_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_2048_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_2048_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_256_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_256_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_256_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_262144_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_262144_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_262144_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_32768_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_32768_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_32768_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_4096_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_4096_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_4096_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_512_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_512_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_512_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_524288_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_524288_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_524288_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_65536_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_65536_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_65536_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReferenceBucket_8192_Dblflt", "1b62814c061a0f80"},
    {"RenderGolden_GpuReferenceBucket_8192_Double", "a590b21058b320d5"},
    {"RenderGolden_GpuReferenceBucket_8192_Float", "922654ed7c75473e"},
    {"RenderGolden_GpuReference_GpuHDRx2x32PerturbedLAv2_View11_Bits32_Store0_Ref10_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "e43583c39e6a9ffe"},
    {"RenderGolden_GpuReference_GpuHDRx2x32PerturbedLAv2_View11_Bits64_Store0_Ref10_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "e43583c39e6a9ffe"},
    {"RenderGolden_GpuReference_GpuHDRx32PerturbedLAv2_View11_Bits32_Store0_Ref10_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8530650d919679d8"},
    {"RenderGolden_GpuReference_GpuHDRx32PerturbedLAv2_View11_Bits64_Store0_Ref10_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "8530650d919679d8"},
    {"RenderGolden_GpuReference_GpuHDRx64PerturbedLAv2_View11_Bits32_Store0_Ref10_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "d6bdd4bb668aa582"},
    {"RenderGolden_GpuReference_GpuHDRx64PerturbedLAv2_View11_Bits64_Store0_Ref10_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "d6bdd4bb668aa582"},
    {"RenderGolden_HardView_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp18_LA1_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_HardView_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp19_LA1_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_HardView_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA1_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_HardView_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp21_LA1_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_HardView_GpuHDRx2x32PerturbedRCLAv2_View27_Bits64_Store0_Ref2_AA1_Step1_Comp22_LA1_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View0.im_Imagina",
     "191a9f91f59489d5"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View0.im_Original",
     "3619b44004dd609b"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_ViewEasy1.im_Imagina",
     "28429533bdced930"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View14.im_Imagina",
     "d7e7706e85183e28"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View14.im_Original",
     "d7e7706e85183e28"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View15_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View15.im_Imagina",
     "4b8974bac4b9d79f"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View15_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View15.im_Original",
     "4b8974bac4b9d79f"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View19_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View19.im_Imagina",
     "5919d354046a8f96"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View19_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View19.im_Original",
     "5919d354046a8f96"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View5.im_Imagina",
     "88184a0787aa341c"},
    {"RenderGolden_Imagina_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View5.im_Original",
     "88184a0787aa341c"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View0.im_Imagina",
     "d6741ef7ec23f823"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View0.im_Original",
     "133514f8b492253c"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_ViewEasy1.im_Imagina",
     "d33dee200e47b6dc"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View14.im_Imagina",
     "3adf30073b4da837"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View14.im_Original",
     "3adf30073b4da837"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View15_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View15.im_Imagina",
     "97757b3bbe15a3ec"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View15_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View15.im_Original",
     "97757b3bbe15a3ec"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View19_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View19.im_Imagina",
     "07fd1bad6e7d90b4"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View19_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View19.im_Original",
     "07fd1bad6e7d90b4"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View5.im_Imagina",
     "991bea00f2bc22fd"},
    {"RenderGolden_Imagina_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_View5.im_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads0_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA1_"
     "Threads0_Load0",
     "0a1bb8834167596c"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA1_"
     "Threads1_Load0",
     "0a1bb8834167596c"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA2_"
     "Threads0_Load0",
     "85c56b126c17d952"},
    {"RenderGolden_LASettings_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA2_"
     "Threads1_Load0",
     "85c56b126c17d952"},
    {"RenderGolden_Legacy_view0-cpu64", "8eee9f298276208d"},
    {"RenderGolden_Legacy_view0-cpu64-aa4", "ccb0a5811b8c13b8"},
    {"RenderGolden_Legacy_view0-cpuhdr", "accf6b90d46ad55e"},
    {"RenderGolden_Legacy_view0-cpuhdr64", "8eee9f298276208d"},
    {"RenderGolden_Legacy_view1-cpu-bla", "408ab08e4feb17f1"},
    {"RenderGolden_Legacy_view5-cpu-bla-v2", "00579b31ca3aa41e"},
    {"RenderGolden_Legacy_view5-cpu-perturbed-bla", "b7221b9cad319c8f"},
    {"RenderGolden_Legacy_view5-cpu32-bla-hdr", "8e2e1d1730c2e8f4"},
    {"RenderGolden_Legacy_view5-cpu32-rc-bla-v2", "3a8313703a7428b6"},
    {"RenderGolden_Legacy_view5-cpu64-bla-hdr", "f64dad356c7780ba"},
    {"RenderGolden_Legacy_view5-cpu64-bla-v2", "88184a0787aa341c"},
    {"RenderGolden_Legacy_view5-cpu64-rc-bla-v2", "2258571d542f1530"},
    {"RenderGolden_OutputPaths_Dotted", "c86d7f6e0cc3ae67"},
    {"RenderGolden_OutputPaths_Extension", "c86d7f6e0cc3ae67"},
    {"RenderGolden_OutputPaths_Unicode", "c86d7f6e0cc3ae67"},
    {"RenderGolden_PerturbedPerturb_Gpu1x64PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_LA0_"
     "Threads1_Load0_HdrOriginal",
     "682dc236b6847859"},
    {"RenderGolden_PerturbedPerturb_Gpu1x64PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_LA0_"
     "Threads1_Load0_HdrPerturbed",
     "874dffe0b36db093"},
    {"RenderGolden_PerturbedPerturb_Gpu1x64PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_HdrOriginal",
     "a0e5c1fe48259c39"},
    {"RenderGolden_PerturbedPerturb_Gpu1x64PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_HdrPerturbed",
     "243613e19e2f739d"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx2x32PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "3d3c8d3abd797329"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx2x32PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Perturbed",
     "09768f6ee087b664"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx2x32PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_"
     "Comp20_LA0_Threads1_Load0_Original",
     "0a290b8c0836a81c"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx2x32PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_"
     "Comp20_LA0_Threads1_Load0_Perturbed",
     "76d85a86b8905c2c"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx32PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "40d26a28bfb5f74b"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx32PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Perturbed",
     "84295ea9f4ebf930"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx32PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "b98812440c0f422c"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx32PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Perturbed",
     "04e60d1e78c35cb1"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx64PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "682dc236b6847859"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx64PerturbedLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Perturbed",
     "874dffe0b36db093"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx64PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "a0e5c1fe48259c39"},
    {"RenderGolden_PerturbedPerturb_GpuHDRx64PerturbedRCLAv2_View14_Bits32_Store0_Ref7_AA4_Step1_Comp20_"
     "LA0_Threads1_Load0_Perturbed",
     "243613e19e2f739d"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "9c640c5cc55d01c8"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "7dab2cbbfe2e5c97"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits32_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "3c6fa656b9f91873"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "9c640c5cc55d01c8"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "7dab2cbbfe2e5c97"},
    {"RenderGolden_Precision_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "3c6fa656b9f91873"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits32_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "55ab2fcd669d051a"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits32_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "f6cbc39c7e6618a4"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits32_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "754dfc2e043ff3ca"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "55ab2fcd669d051a"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "9b6046d9985b1311"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "f6cbc39c7e6618a4"},
    {"RenderGolden_Precision_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "754dfc2e043ff3ca"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits32_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "21b45da72ed3b31e"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits32_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "97416b80e84db0d2"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits32_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "4d1c685a5baf7774"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "21b45da72ed3b31e"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "6de457bad948b2b2"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "97416b80e84db0d2"},
    {"RenderGolden_Precision_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "4d1c685a5baf7774"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits32_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "3f284c7693b7ccf0"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits32_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits32_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "972c9961c998e0f9"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits32_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "7ff6acbff23743bd"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step16_Comp20_LA0_Threads1_Load0",
     "3f284c7693b7ccf0"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0",
     "7239f57e08d37a79"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step4_Comp20_LA0_Threads1_Load0",
     "972c9961c998e0f9"},
    {"RenderGolden_Precision_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step8_Comp20_LA0_Threads1_Load0",
     "7ff6acbff23743bd"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref0_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "331fb878ae2fc4ce"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref10_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "f54bf88bfb103b50"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref11_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref1_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "331fb878ae2fc4ce"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref3_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref4_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref4_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref5_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref5_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref6_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref6_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref7_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref7_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref8_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits32_Store0_Ref9_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref0_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "331fb878ae2fc4ce"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref10_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "f54bf88bfb103b50"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref11_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref1_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "331fb878ae2fc4ce"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref3_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref4_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref4_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref5_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref5_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref6_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref6_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref7_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref7_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_ReusedAfterZoom",
     "1c574576a8406fd3"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref8_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceBackend_Cpu64PerturbedBLAV2HDR_View5_Bits64_Store0_Ref9_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "88184a0787aa341c"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Decompressed",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Original",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Reset",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Original",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_ReferenceSave_Gpu1x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Reset",
     "ad1e24a1ba6ae5e8"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "f76cb61610eb1f8d"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "f181b53898cd0e2e"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "c070612cfacbdc58"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "a1af5bda205c87ec"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "1fa413faf41d1adb"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "f181b53898cd0e2e"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "1fa413faf41d1adb"},
    {"RenderGolden_ReferenceSave_Gpu1x64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Decompressed",
     "9b6046d9985b1311"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Original",
     "9b6046d9985b1311"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Reset",
     "9b6046d9985b1311"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Original",
     "9b6046d9985b1311"},
    {"RenderGolden_ReferenceSave_Gpu1x64_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Reset",
     "9b6046d9985b1311"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Decompressed",
     "6de457bad948b2b2"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Original",
     "6de457bad948b2b2"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Reset",
     "6de457bad948b2b2"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Original",
     "6de457bad948b2b2"},
    {"RenderGolden_ReferenceSave_Gpu2x32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Reset",
     "6de457bad948b2b2"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "81e6d6004ab25a85"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "4252cdd30991636c"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "4252cdd30991636c"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "d22556a0dfa18151"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "9ee5687d10eea487"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "9ee5687d10eea487"},
    {"RenderGolden_ReferenceSave_GpuHDRx2x32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "e1084fb00a5cbaa8"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "c225b8b2bdd35d59"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "95a118b8e30625dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "659f5b51ed833a15"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "95a118b8e30625dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "519bbd95e6ad3591"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "f09dce055b6a385f"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "16c1b4d57b3979dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "f09dce055b6a385f"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "199f0e66b9cc8997"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "b15380b9fe208644"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "b15380b9fe208644"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "e082d6894bf6f275"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "8224dd1d7eb95233"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "659f5b51ed833a15"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "8224dd1d7eb95233"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "09f69837e292558d"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "64b788b1d1c59eed"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "16c1b4d57b3979dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "64b788b1d1c59eed"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "b0c8dc1ebe0e749f"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "bb26bab1e662bb63"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "bb26bab1e662bb63"},
    {"RenderGolden_ReferenceSave_GpuHDRx32PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Decompressed",
     "7239f57e08d37a79"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Original",
     "7239f57e08d37a79"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load0_"
     "Reset",
     "7239f57e08d37a79"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Decompressed",
     "2547a04ce83b3c97"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Original",
     "7239f57e08d37a79"},
    {"RenderGolden_ReferenceSave_GpuHDRx32_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_Threads1_Load1_"
     "Reset",
     "7239f57e08d37a79"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "6891501a57e1e456"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "35630b5b9a8a03ca"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "659f5b51ed833a15"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "35630b5b9a8a03ca"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "7cab0ae73785379e"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "3adf30073b4da837"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "16c1b4d57b3979dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "3adf30073b4da837"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "de569a2881ec0dec"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "991bea00f2bc22fd"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View0_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View10_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1",
     "INCOMPLETE"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "3419e344827e1b90"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "8f14d08adb5e6342"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "659f5b51ed833a15"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "8f14d08adb5e6342"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View13_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Decompressed",
     "54c393cf45ffc855"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Original",
     "7fa3ffca39df1162"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Decompressed",
     "16c1b4d57b3979dd"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Original",
     "7fa3ffca39df1162"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View14_Bits64_Store0_Ref2_AA1_Step1_Comp20_"
     "LA0_Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Decompressed",
     "389a50fd8c20f3c4"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Original",
     "34332c958151daae"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load0_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Decompressed",
     "01a2a484e3612506"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Original",
     "34332c958151daae"},
    {"RenderGolden_ReferenceSave_GpuHDRx64PerturbedRCLAv2_View5_Bits64_Store0_Ref2_AA1_Step1_Comp20_LA0_"
     "Threads1_Load1_Reset",
     "133514f8b492253c"},
    {"RenderGolden_ResizeCpu_Frame0", "60d0a5d1515a6f23"},
    {"RenderGolden_ResizeCpu_Frame1", "31f51e77cfc84dcf"},
    {"RenderGolden_ResizeCpu_Frame10", "b20bbae2d53b4b6c"},
    {"RenderGolden_ResizeCpu_Frame100", "fca7240ce7c8890f"},
    {"RenderGolden_ResizeCpu_Frame101", "920058b29d49425c"},
    {"RenderGolden_ResizeCpu_Frame102", "b0b48ef674ce271b"},
    {"RenderGolden_ResizeCpu_Frame103", "2185fe093a1e2a8c"},
    {"RenderGolden_ResizeCpu_Frame104", "5f845a652f9354d3"},
    {"RenderGolden_ResizeCpu_Frame105", "c69c543a800d0f13"},
    {"RenderGolden_ResizeCpu_Frame106", "dc99dfd2071de851"},
    {"RenderGolden_ResizeCpu_Frame107", "5180cb901ec91075"},
    {"RenderGolden_ResizeCpu_Frame108", "74237c060daca3ca"},
    {"RenderGolden_ResizeCpu_Frame109", "7975b95782ed2413"},
    {"RenderGolden_ResizeCpu_Frame11", "c1bfe8b68cd76467"},
    {"RenderGolden_ResizeCpu_Frame110", "397d4b60ff746edc"},
    {"RenderGolden_ResizeCpu_Frame111", "b588364792286076"},
    {"RenderGolden_ResizeCpu_Frame112", "aab87926783dec73"},
    {"RenderGolden_ResizeCpu_Frame113", "cd3d81f990aa8ad2"},
    {"RenderGolden_ResizeCpu_Frame114", "0682b44425f6eb97"},
    {"RenderGolden_ResizeCpu_Frame115", "781afe80e198cbe0"},
    {"RenderGolden_ResizeCpu_Frame116", "6f87e7584175237a"},
    {"RenderGolden_ResizeCpu_Frame117", "377e15c8affa8ec8"},
    {"RenderGolden_ResizeCpu_Frame118", "483e2caf3b1261cc"},
    {"RenderGolden_ResizeCpu_Frame119", "fc3723c0b4284edb"},
    {"RenderGolden_ResizeCpu_Frame12", "89ffdf5eb173f7ae"},
    {"RenderGolden_ResizeCpu_Frame120", "7d728be8c8c1105b"},
    {"RenderGolden_ResizeCpu_Frame121", "425e1925fd1b4440"},
    {"RenderGolden_ResizeCpu_Frame122", "5de4f4f83760c355"},
    {"RenderGolden_ResizeCpu_Frame123", "0103efbe7b4eedb8"},
    {"RenderGolden_ResizeCpu_Frame124", "d216c09ac55fe396"},
    {"RenderGolden_ResizeCpu_Frame125", "09dc00ec254b7342"},
    {"RenderGolden_ResizeCpu_Frame126", "b8cdfe52d912be3b"},
    {"RenderGolden_ResizeCpu_Frame127", "7155106d9bcde200"},
    {"RenderGolden_ResizeCpu_Frame128", "6ecd2e6f40d8123c"},
    {"RenderGolden_ResizeCpu_Frame129", "e7ea31c657882ef1"},
    {"RenderGolden_ResizeCpu_Frame13", "04c1f6497108a83e"},
    {"RenderGolden_ResizeCpu_Frame130", "00d5a3d5ef6ea760"},
    {"RenderGolden_ResizeCpu_Frame131", "09994db4aa68a89e"},
    {"RenderGolden_ResizeCpu_Frame132", "43409aa289304952"},
    {"RenderGolden_ResizeCpu_Frame133", "4d6746874d90a372"},
    {"RenderGolden_ResizeCpu_Frame134", "afa7d1e39cc9e72c"},
    {"RenderGolden_ResizeCpu_Frame135", "db3e61ca50d9afcc"},
    {"RenderGolden_ResizeCpu_Frame136", "b31792648a53dfbc"},
    {"RenderGolden_ResizeCpu_Frame137", "60a99cc9a9aade4a"},
    {"RenderGolden_ResizeCpu_Frame138", "d13b6f255700a470"},
    {"RenderGolden_ResizeCpu_Frame139", "b4c85c3dd1400de7"},
    {"RenderGolden_ResizeCpu_Frame14", "67903335137b48f8"},
    {"RenderGolden_ResizeCpu_Frame140", "8f535f873eb7d0cd"},
    {"RenderGolden_ResizeCpu_Frame141", "bc1862ee6ea70543"},
    {"RenderGolden_ResizeCpu_Frame142", "6bedac818ddde9e0"},
    {"RenderGolden_ResizeCpu_Frame143", "1698637ad512ee2b"},
    {"RenderGolden_ResizeCpu_Frame144", "28c0546acf7705a1"},
    {"RenderGolden_ResizeCpu_Frame145", "82f19c9fbe827da3"},
    {"RenderGolden_ResizeCpu_Frame146", "f01451676cd2da0b"},
    {"RenderGolden_ResizeCpu_Frame147", "84795db55e677125"},
    {"RenderGolden_ResizeCpu_Frame148", "5409605c1f446d0a"},
    {"RenderGolden_ResizeCpu_Frame149", "129b8980b9790b2e"},
    {"RenderGolden_ResizeCpu_Frame15", "3366d5656e019659"},
    {"RenderGolden_ResizeCpu_Frame150", "a1e3386995994104"},
    {"RenderGolden_ResizeCpu_Frame151", "b290f246a3cbd223"},
    {"RenderGolden_ResizeCpu_Frame152", "2c42db9c74efd3b4"},
    {"RenderGolden_ResizeCpu_Frame153", "7ba07ff57a0251c9"},
    {"RenderGolden_ResizeCpu_Frame154", "10332e46791eeec7"},
    {"RenderGolden_ResizeCpu_Frame155", "6dee26738f2002a7"},
    {"RenderGolden_ResizeCpu_Frame156", "e3305af4fdca71a8"},
    {"RenderGolden_ResizeCpu_Frame157", "3f765f60671325e1"},
    {"RenderGolden_ResizeCpu_Frame158", "ea39fb59b8c0a661"},
    {"RenderGolden_ResizeCpu_Frame159", "c9bccd9b43c42482"},
    {"RenderGolden_ResizeCpu_Frame16", "af80ad52aa77cdc5"},
    {"RenderGolden_ResizeCpu_Frame160", "28d53ef7e20c0160"},
    {"RenderGolden_ResizeCpu_Frame161", "7a9d57e2afaf37d6"},
    {"RenderGolden_ResizeCpu_Frame162", "a5b4ecf9b39df02b"},
    {"RenderGolden_ResizeCpu_Frame163", "7eaded0ac229ec0f"},
    {"RenderGolden_ResizeCpu_Frame164", "6bc2888683eee285"},
    {"RenderGolden_ResizeCpu_Frame165", "146c78a15e2e8cdd"},
    {"RenderGolden_ResizeCpu_Frame166", "84d6e46b0c1942b6"},
    {"RenderGolden_ResizeCpu_Frame167", "ea0ffad32f33c071"},
    {"RenderGolden_ResizeCpu_Frame168", "135d57e410a2c786"},
    {"RenderGolden_ResizeCpu_Frame169", "3eb592d7f735f5a4"},
    {"RenderGolden_ResizeCpu_Frame17", "db8d3a33b8e73fc6"},
    {"RenderGolden_ResizeCpu_Frame170", "6b18a93b47b01164"},
    {"RenderGolden_ResizeCpu_Frame171", "927c4d60f81af537"},
    {"RenderGolden_ResizeCpu_Frame172", "5971b1f6b8e17351"},
    {"RenderGolden_ResizeCpu_Frame173", "a33f784a19df33b9"},
    {"RenderGolden_ResizeCpu_Frame174", "9d836254b549950a"},
    {"RenderGolden_ResizeCpu_Frame175", "c508e4086f154410"},
    {"RenderGolden_ResizeCpu_Frame176", "d75dcc0d87dd77c1"},
    {"RenderGolden_ResizeCpu_Frame177", "56df1ffb7b955736"},
    {"RenderGolden_ResizeCpu_Frame178", "2b5064237709b100"},
    {"RenderGolden_ResizeCpu_Frame179", "1e10b0d6b1ff1ac1"},
    {"RenderGolden_ResizeCpu_Frame18", "60aaa7b27c630e99"},
    {"RenderGolden_ResizeCpu_Frame180", "6dd274fee036126a"},
    {"RenderGolden_ResizeCpu_Frame181", "eb40a14bfdf8ebaa"},
    {"RenderGolden_ResizeCpu_Frame182", "219a1b0a44d1afc4"},
    {"RenderGolden_ResizeCpu_Frame183", "6379007b337971cf"},
    {"RenderGolden_ResizeCpu_Frame184", "b8f12855217d6944"},
    {"RenderGolden_ResizeCpu_Frame185", "9726f041047ddcab"},
    {"RenderGolden_ResizeCpu_Frame186", "d70a17d8b3bd5400"},
    {"RenderGolden_ResizeCpu_Frame187", "938e3753b8d819c1"},
    {"RenderGolden_ResizeCpu_Frame188", "d4b4d39d877add40"},
    {"RenderGolden_ResizeCpu_Frame189", "8fa272c811e140cf"},
    {"RenderGolden_ResizeCpu_Frame19", "bcc88f1ace92f7ac"},
    {"RenderGolden_ResizeCpu_Frame190", "e62c983f69b7013f"},
    {"RenderGolden_ResizeCpu_Frame191", "442ffc514437e660"},
    {"RenderGolden_ResizeCpu_Frame192", "c00cdd6c8f111bb4"},
    {"RenderGolden_ResizeCpu_Frame193", "833646c2968cb71e"},
    {"RenderGolden_ResizeCpu_Frame194", "f5dc059d3013da2d"},
    {"RenderGolden_ResizeCpu_Frame195", "a6edc6cddbcb79d2"},
    {"RenderGolden_ResizeCpu_Frame196", "e041392fc2c4aa10"},
    {"RenderGolden_ResizeCpu_Frame197", "47f8e4c5f86c3729"},
    {"RenderGolden_ResizeCpu_Frame198", "acb6655ed42a8584"},
    {"RenderGolden_ResizeCpu_Frame199", "936cd4320c0e647a"},
    {"RenderGolden_ResizeCpu_Frame2", "4c5ef5e0c5858bd6"},
    {"RenderGolden_ResizeCpu_Frame20", "448d80817e980bfc"},
    {"RenderGolden_ResizeCpu_Frame200", "f813476d44198d2a"},
    {"RenderGolden_ResizeCpu_Frame201", "e9ea27b3198587d2"},
    {"RenderGolden_ResizeCpu_Frame202", "d2044dcdae81f5f5"},
    {"RenderGolden_ResizeCpu_Frame203", "1806c1a40c99883a"},
    {"RenderGolden_ResizeCpu_Frame204", "4a542b6586d129c2"},
    {"RenderGolden_ResizeCpu_Frame205", "06cb9ee2fff8c5d1"},
    {"RenderGolden_ResizeCpu_Frame206", "f90fb6a4377c75ef"},
    {"RenderGolden_ResizeCpu_Frame207", "29137bde9c0f66d4"},
    {"RenderGolden_ResizeCpu_Frame208", "89f5a25a85d9da02"},
    {"RenderGolden_ResizeCpu_Frame209", "610e47046618abcc"},
    {"RenderGolden_ResizeCpu_Frame21", "42c5ab5b117b84e1"},
    {"RenderGolden_ResizeCpu_Frame210", "ebbc292360873ca7"},
    {"RenderGolden_ResizeCpu_Frame211", "fb4647c119b78389"},
    {"RenderGolden_ResizeCpu_Frame212", "5aa5ca470509fc8c"},
    {"RenderGolden_ResizeCpu_Frame213", "1163a11b1c3bfc7c"},
    {"RenderGolden_ResizeCpu_Frame214", "ae58ab94eda0db05"},
    {"RenderGolden_ResizeCpu_Frame215", "6eb30938f262689e"},
    {"RenderGolden_ResizeCpu_Frame216", "715a65174ea53c33"},
    {"RenderGolden_ResizeCpu_Frame217", "98b2d373738dee31"},
    {"RenderGolden_ResizeCpu_Frame218", "5c44d2fd944eeae2"},
    {"RenderGolden_ResizeCpu_Frame219", "2fd778d15428b3cd"},
    {"RenderGolden_ResizeCpu_Frame22", "122eefd142badd93"},
    {"RenderGolden_ResizeCpu_Frame220", "898ce786dc4e184a"},
    {"RenderGolden_ResizeCpu_Frame221", "79de11c5f46dd058"},
    {"RenderGolden_ResizeCpu_Frame222", "aa4e2a4f6522d2ea"},
    {"RenderGolden_ResizeCpu_Frame223", "07f9c60f411a9582"},
    {"RenderGolden_ResizeCpu_Frame224", "3e3ec6bf978055fc"},
    {"RenderGolden_ResizeCpu_Frame225", "f8589a5a0b871fcc"},
    {"RenderGolden_ResizeCpu_Frame226", "e367e713a51b94f9"},
    {"RenderGolden_ResizeCpu_Frame227", "f439cf3305788f40"},
    {"RenderGolden_ResizeCpu_Frame228", "819fed288036aac0"},
    {"RenderGolden_ResizeCpu_Frame229", "8fe802e4774857d2"},
    {"RenderGolden_ResizeCpu_Frame23", "346fd63cd826078b"},
    {"RenderGolden_ResizeCpu_Frame230", "45a5476fe08ab146"},
    {"RenderGolden_ResizeCpu_Frame231", "92c8461d6da664cd"},
    {"RenderGolden_ResizeCpu_Frame232", "bd75f669d1483f3c"},
    {"RenderGolden_ResizeCpu_Frame233", "da40d85b57ba3356"},
    {"RenderGolden_ResizeCpu_Frame234", "9be7f0509a2ff10e"},
    {"RenderGolden_ResizeCpu_Frame235", "f40a5db921494aff"},
    {"RenderGolden_ResizeCpu_Frame236", "c342c372df96e55f"},
    {"RenderGolden_ResizeCpu_Frame237", "3ffef20fa47dce74"},
    {"RenderGolden_ResizeCpu_Frame238", "b9b08d84073dad24"},
    {"RenderGolden_ResizeCpu_Frame239", "f8f6cc8e003fee5b"},
    {"RenderGolden_ResizeCpu_Frame24", "8b04f55ba2c49eb6"},
    {"RenderGolden_ResizeCpu_Frame240", "7f4fcdf522423dd3"},
    {"RenderGolden_ResizeCpu_Frame241", "ee59a012d757e8de"},
    {"RenderGolden_ResizeCpu_Frame242", "c106df27c5567b3a"},
    {"RenderGolden_ResizeCpu_Frame243", "229d0a97bd3c846d"},
    {"RenderGolden_ResizeCpu_Frame244", "ba485b2869cf8160"},
    {"RenderGolden_ResizeCpu_Frame245", "0a947ed7385cc469"},
    {"RenderGolden_ResizeCpu_Frame246", "081164ecb8c7fdc6"},
    {"RenderGolden_ResizeCpu_Frame247", "63b23aea637f096f"},
    {"RenderGolden_ResizeCpu_Frame248", "c4bf2e2adb8e9b0a"},
    {"RenderGolden_ResizeCpu_Frame249", "fbf17cebe662f54a"},
    {"RenderGolden_ResizeCpu_Frame25", "c12cbd0102412624"},
    {"RenderGolden_ResizeCpu_Frame250", "9089c3c0ae41a5b5"},
    {"RenderGolden_ResizeCpu_Frame251", "96f05eb45d99e5b6"},
    {"RenderGolden_ResizeCpu_Frame252", "8b682bf84e1fb134"},
    {"RenderGolden_ResizeCpu_Frame253", "c20c27fb2a9e5be8"},
    {"RenderGolden_ResizeCpu_Frame254", "17966e943b5f8ecb"},
    {"RenderGolden_ResizeCpu_Frame255", "50401991a1e9d07c"},
    {"RenderGolden_ResizeCpu_Frame256", "a8e019c7459ae9ed"},
    {"RenderGolden_ResizeCpu_Frame257", "32427e6dc7e54087"},
    {"RenderGolden_ResizeCpu_Frame258", "333e75ba7daf3976"},
    {"RenderGolden_ResizeCpu_Frame259", "607c47e318bab7b0"},
    {"RenderGolden_ResizeCpu_Frame26", "17a84d811b6cbbf6"},
    {"RenderGolden_ResizeCpu_Frame260", "4c0145ebce14da12"},
    {"RenderGolden_ResizeCpu_Frame261", "5f8823a9e4496276"},
    {"RenderGolden_ResizeCpu_Frame262", "a0252d99803e9a5e"},
    {"RenderGolden_ResizeCpu_Frame263", "a4c7c142c8008141"},
    {"RenderGolden_ResizeCpu_Frame264", "2a5d15954230ecd2"},
    {"RenderGolden_ResizeCpu_Frame265", "abfaabae1d883643"},
    {"RenderGolden_ResizeCpu_Frame266", "03bca85e8b949dac"},
    {"RenderGolden_ResizeCpu_Frame267", "81db44705a8cf9bf"},
    {"RenderGolden_ResizeCpu_Frame268", "92257552af91c211"},
    {"RenderGolden_ResizeCpu_Frame269", "bcbb05fba5ac56ae"},
    {"RenderGolden_ResizeCpu_Frame27", "fdb363877f988b81"},
    {"RenderGolden_ResizeCpu_Frame270", "7214874573855c46"},
    {"RenderGolden_ResizeCpu_Frame271", "9c07b9a08a51bd58"},
    {"RenderGolden_ResizeCpu_Frame272", "f9cf473ee9dd05c9"},
    {"RenderGolden_ResizeCpu_Frame273", "d7a2fb6cfc9faa20"},
    {"RenderGolden_ResizeCpu_Frame274", "cc0d048bfc05cfd6"},
    {"RenderGolden_ResizeCpu_Frame275", "b81b7e765f7dd78e"},
    {"RenderGolden_ResizeCpu_Frame276", "82885d1f653fa7e8"},
    {"RenderGolden_ResizeCpu_Frame277", "1d090474c0c2227b"},
    {"RenderGolden_ResizeCpu_Frame278", "86738c32375457e3"},
    {"RenderGolden_ResizeCpu_Frame279", "fb77f51f90d798a0"},
    {"RenderGolden_ResizeCpu_Frame28", "466c1d7614fcb232"},
    {"RenderGolden_ResizeCpu_Frame280", "0f1fab75979bf5ed"},
    {"RenderGolden_ResizeCpu_Frame281", "6438b2d074575062"},
    {"RenderGolden_ResizeCpu_Frame282", "9924f9150246e2c9"},
    {"RenderGolden_ResizeCpu_Frame283", "f9f2db5159c88b20"},
    {"RenderGolden_ResizeCpu_Frame284", "8df439dd95452087"},
    {"RenderGolden_ResizeCpu_Frame285", "f0c2aecca8c5ab21"},
    {"RenderGolden_ResizeCpu_Frame286", "2269c025712372f6"},
    {"RenderGolden_ResizeCpu_Frame287", "8dec2b81b2d4f878"},
    {"RenderGolden_ResizeCpu_Frame288", "cb8caa899c8fd2d2"},
    {"RenderGolden_ResizeCpu_Frame289", "6ad7659db2e010aa"},
    {"RenderGolden_ResizeCpu_Frame29", "16bd86a5f2c34c10"},
    {"RenderGolden_ResizeCpu_Frame290", "0a4354d032aa624a"},
    {"RenderGolden_ResizeCpu_Frame291", "f76da0e42599b854"},
    {"RenderGolden_ResizeCpu_Frame292", "a238883e481018fd"},
    {"RenderGolden_ResizeCpu_Frame293", "884fcf8304156d26"},
    {"RenderGolden_ResizeCpu_Frame294", "0099d021514f3e94"},
    {"RenderGolden_ResizeCpu_Frame295", "e8dbb79510255d5e"},
    {"RenderGolden_ResizeCpu_Frame296", "4a025fa5bd37e339"},
    {"RenderGolden_ResizeCpu_Frame297", "3b5dc382a57b737c"},
    {"RenderGolden_ResizeCpu_Frame298", "4e5ff0cadd9c5809"},
    {"RenderGolden_ResizeCpu_Frame299", "92f440ea3792c036"},
    {"RenderGolden_ResizeCpu_Frame3", "d7328adccc2777fb"},
    {"RenderGolden_ResizeCpu_Frame30", "e6bd43235419eb7d"},
    {"RenderGolden_ResizeCpu_Frame300", "c368adab263d639d"},
    {"RenderGolden_ResizeCpu_Frame301", "901705d38f490483"},
    {"RenderGolden_ResizeCpu_Frame302", "05a9b19edb6663d0"},
    {"RenderGolden_ResizeCpu_Frame303", "7f66ff1084729d13"},
    {"RenderGolden_ResizeCpu_Frame304", "0c4ce8f8a062a6e3"},
    {"RenderGolden_ResizeCpu_Frame305", "ef42731d6fa1ea7b"},
    {"RenderGolden_ResizeCpu_Frame306", "b0288e63c6dc8ff2"},
    {"RenderGolden_ResizeCpu_Frame307", "7db4baf91e4674e8"},
    {"RenderGolden_ResizeCpu_Frame308", "dae4cf5c66ee343e"},
    {"RenderGolden_ResizeCpu_Frame309", "9fa444add1171e8e"},
    {"RenderGolden_ResizeCpu_Frame31", "46c968e599d2f73b"},
    {"RenderGolden_ResizeCpu_Frame310", "358fb3e63cc8135f"},
    {"RenderGolden_ResizeCpu_Frame311", "9590b406e6180dc6"},
    {"RenderGolden_ResizeCpu_Frame312", "b3645d8d6b56ec17"},
    {"RenderGolden_ResizeCpu_Frame313", "9edc76740bca048c"},
    {"RenderGolden_ResizeCpu_Frame314", "9ad79efb39fa2d8d"},
    {"RenderGolden_ResizeCpu_Frame315", "b322a5de54dbbc2f"},
    {"RenderGolden_ResizeCpu_Frame316", "d079c9bcc5c0a917"},
    {"RenderGolden_ResizeCpu_Frame317", "93655300aed98c77"},
    {"RenderGolden_ResizeCpu_Frame318", "28971a260cca89e3"},
    {"RenderGolden_ResizeCpu_Frame319", "0d628bb824c5703a"},
    {"RenderGolden_ResizeCpu_Frame32", "a409c63a0dc638ff"},
    {"RenderGolden_ResizeCpu_Frame320", "f4f8d55ec92fa14f"},
    {"RenderGolden_ResizeCpu_Frame321", "d15202a86689a438"},
    {"RenderGolden_ResizeCpu_Frame322", "ff34c4d84bcb8445"},
    {"RenderGolden_ResizeCpu_Frame323", "f5341285d8d079c9"},
    {"RenderGolden_ResizeCpu_Frame324", "23a285fc82f665b3"},
    {"RenderGolden_ResizeCpu_Frame325", "0bcb5c6bae378af8"},
    {"RenderGolden_ResizeCpu_Frame326", "f2137967410c7204"},
    {"RenderGolden_ResizeCpu_Frame327", "706b6f7034a8a391"},
    {"RenderGolden_ResizeCpu_Frame328", "4fcaaafb25055ebc"},
    {"RenderGolden_ResizeCpu_Frame329", "2507f487d2d97309"},
    {"RenderGolden_ResizeCpu_Frame33", "ede272f53595198c"},
    {"RenderGolden_ResizeCpu_Frame330", "32369cf59589f224"},
    {"RenderGolden_ResizeCpu_Frame331", "7e46d9b84da9f07b"},
    {"RenderGolden_ResizeCpu_Frame332", "86d8d787d156f001"},
    {"RenderGolden_ResizeCpu_Frame333", "abea0971d7b07150"},
    {"RenderGolden_ResizeCpu_Frame334", "20ca8ff6e598af4d"},
    {"RenderGolden_ResizeCpu_Frame335", "4baf81d56646c952"},
    {"RenderGolden_ResizeCpu_Frame336", "7bc01f148661a044"},
    {"RenderGolden_ResizeCpu_Frame337", "822c7dae096aaec2"},
    {"RenderGolden_ResizeCpu_Frame338", "498d2fac2d018469"},
    {"RenderGolden_ResizeCpu_Frame339", "ac7b1cae9ce5a6fc"},
    {"RenderGolden_ResizeCpu_Frame34", "3fadf51997701165"},
    {"RenderGolden_ResizeCpu_Frame340", "1d3534720d47cf48"},
    {"RenderGolden_ResizeCpu_Frame341", "f23d9a974f44ca12"},
    {"RenderGolden_ResizeCpu_Frame342", "6c2ce39f8f086e67"},
    {"RenderGolden_ResizeCpu_Frame343", "20566af9d0c49b87"},
    {"RenderGolden_ResizeCpu_Frame344", "6129758e7341776a"},
    {"RenderGolden_ResizeCpu_Frame345", "ea63b3b28bd6f203"},
    {"RenderGolden_ResizeCpu_Frame346", "db5c352aa2aabaa3"},
    {"RenderGolden_ResizeCpu_Frame347", "ad89f66abc7797dc"},
    {"RenderGolden_ResizeCpu_Frame348", "42d4ae09136b3c4e"},
    {"RenderGolden_ResizeCpu_Frame349", "81c01f5748783fc2"},
    {"RenderGolden_ResizeCpu_Frame35", "d617785adf6adb23"},
    {"RenderGolden_ResizeCpu_Frame350", "a3446da5a39a9df0"},
    {"RenderGolden_ResizeCpu_Frame351", "78ee3bec7eb96403"},
    {"RenderGolden_ResizeCpu_Frame352", "d3a404d077e62f1f"},
    {"RenderGolden_ResizeCpu_Frame353", "0e88b4677dd1a0db"},
    {"RenderGolden_ResizeCpu_Frame354", "085b4a53a343efe7"},
    {"RenderGolden_ResizeCpu_Frame355", "2743c177244ccdf6"},
    {"RenderGolden_ResizeCpu_Frame356", "5f1187d243778999"},
    {"RenderGolden_ResizeCpu_Frame357", "d1f827c08e9aa309"},
    {"RenderGolden_ResizeCpu_Frame358", "613d5c68d17b0672"},
    {"RenderGolden_ResizeCpu_Frame359", "e9d81202c669e6be"},
    {"RenderGolden_ResizeCpu_Frame36", "9033741515b57c9e"},
    {"RenderGolden_ResizeCpu_Frame360", "4ca393c83f1e7f12"},
    {"RenderGolden_ResizeCpu_Frame361", "39106f112ec7e2f7"},
    {"RenderGolden_ResizeCpu_Frame362", "23963109c421f5ba"},
    {"RenderGolden_ResizeCpu_Frame363", "4c9018baa9ad1677"},
    {"RenderGolden_ResizeCpu_Frame364", "71698310da5c53cc"},
    {"RenderGolden_ResizeCpu_Frame365", "c7daf4eec2fe9c6d"},
    {"RenderGolden_ResizeCpu_Frame366", "52b83fa34087cab5"},
    {"RenderGolden_ResizeCpu_Frame367", "0763ee127a60c9e0"},
    {"RenderGolden_ResizeCpu_Frame368", "b80fa69e70e38dc8"},
    {"RenderGolden_ResizeCpu_Frame369", "0df2012f1cae97dd"},
    {"RenderGolden_ResizeCpu_Frame37", "f1b3c2797b9d4374"},
    {"RenderGolden_ResizeCpu_Frame370", "fa5466b8d57b3d2e"},
    {"RenderGolden_ResizeCpu_Frame371", "3c9824777cfe68f9"},
    {"RenderGolden_ResizeCpu_Frame372", "c62c1511b5793d1b"},
    {"RenderGolden_ResizeCpu_Frame373", "02b607401a6f6370"},
    {"RenderGolden_ResizeCpu_Frame374", "cb9b7d747b51f482"},
    {"RenderGolden_ResizeCpu_Frame375", "72625c25111b5dfe"},
    {"RenderGolden_ResizeCpu_Frame376", "223328e332ffba26"},
    {"RenderGolden_ResizeCpu_Frame377", "c41fe01da4b325f5"},
    {"RenderGolden_ResizeCpu_Frame378", "9f53149de2ab8a53"},
    {"RenderGolden_ResizeCpu_Frame379", "8a750ab926c16deb"},
    {"RenderGolden_ResizeCpu_Frame38", "178df3009afb4cbc"},
    {"RenderGolden_ResizeCpu_Frame380", "9ddff0922f14801e"},
    {"RenderGolden_ResizeCpu_Frame381", "2726a8e65a143efb"},
    {"RenderGolden_ResizeCpu_Frame382", "189aac62fe89e305"},
    {"RenderGolden_ResizeCpu_Frame383", "3c87eb36506362d0"},
    {"RenderGolden_ResizeCpu_Frame384", "9e2184fb08a96bdf"},
    {"RenderGolden_ResizeCpu_Frame385", "89d472009ba8fab5"},
    {"RenderGolden_ResizeCpu_Frame386", "c8346ab9b3441c07"},
    {"RenderGolden_ResizeCpu_Frame387", "f02f87b326cbc15a"},
    {"RenderGolden_ResizeCpu_Frame388", "8f512be5bb1ca163"},
    {"RenderGolden_ResizeCpu_Frame389", "a630e5af6d7e1eb9"},
    {"RenderGolden_ResizeCpu_Frame39", "d219b4959db52982"},
    {"RenderGolden_ResizeCpu_Frame390", "4e87aada0a1ea624"},
    {"RenderGolden_ResizeCpu_Frame391", "3f4a677ae935e657"},
    {"RenderGolden_ResizeCpu_Frame392", "aee68be3500b6659"},
    {"RenderGolden_ResizeCpu_Frame393", "a79c20e832f32897"},
    {"RenderGolden_ResizeCpu_Frame394", "0b31b09d54a2be77"},
    {"RenderGolden_ResizeCpu_Frame395", "09e89095cbbff605"},
    {"RenderGolden_ResizeCpu_Frame396", "53228b675f0e504c"},
    {"RenderGolden_ResizeCpu_Frame397", "9c2788e2642b4ae2"},
    {"RenderGolden_ResizeCpu_Frame398", "52e11dc9bda32fd4"},
    {"RenderGolden_ResizeCpu_Frame399", "8728132197b7a691"},
    {"RenderGolden_ResizeCpu_Frame4", "09f1e7e7d31ca890"},
    {"RenderGolden_ResizeCpu_Frame40", "772547dd076f2b40"},
    {"RenderGolden_ResizeCpu_Frame400", "35c467a54cfa75d3"},
    {"RenderGolden_ResizeCpu_Frame401", "e8ac66c3351c3971"},
    {"RenderGolden_ResizeCpu_Frame402", "ee2d0a6bded3f6cf"},
    {"RenderGolden_ResizeCpu_Frame403", "7bbaf67cf771e151"},
    {"RenderGolden_ResizeCpu_Frame404", "06186f32d2e3360d"},
    {"RenderGolden_ResizeCpu_Frame405", "00900fa58750d6ce"},
    {"RenderGolden_ResizeCpu_Frame406", "d49dcb313fb2a8d5"},
    {"RenderGolden_ResizeCpu_Frame407", "271cc8d76ce742c5"},
    {"RenderGolden_ResizeCpu_Frame408", "88b5e068f7962d4e"},
    {"RenderGolden_ResizeCpu_Frame409", "d8e7c044eb9f9139"},
    {"RenderGolden_ResizeCpu_Frame41", "0db67eb186d3ee8c"},
    {"RenderGolden_ResizeCpu_Frame410", "7ef4ab2435e622a1"},
    {"RenderGolden_ResizeCpu_Frame411", "bef7d413f5e62b1d"},
    {"RenderGolden_ResizeCpu_Frame412", "4a8c980b73fd6373"},
    {"RenderGolden_ResizeCpu_Frame413", "bab18dc73c0dad8c"},
    {"RenderGolden_ResizeCpu_Frame414", "f1e3109cf51aa93a"},
    {"RenderGolden_ResizeCpu_Frame415", "15eda191642ccf2e"},
    {"RenderGolden_ResizeCpu_Frame416", "eb6d198dcd0cba98"},
    {"RenderGolden_ResizeCpu_Frame417", "6d3056b19b33d37d"},
    {"RenderGolden_ResizeCpu_Frame418", "5da4059aa4ff645a"},
    {"RenderGolden_ResizeCpu_Frame419", "644ad1589197116b"},
    {"RenderGolden_ResizeCpu_Frame42", "627afa7ec3bc4afe"},
    {"RenderGolden_ResizeCpu_Frame420", "fd45dd3701195160"},
    {"RenderGolden_ResizeCpu_Frame421", "50cace3cff5716ae"},
    {"RenderGolden_ResizeCpu_Frame422", "7e389701a7388596"},
    {"RenderGolden_ResizeCpu_Frame423", "f8c731f73fc0c477"},
    {"RenderGolden_ResizeCpu_Frame424", "11d4031d1de04631"},
    {"RenderGolden_ResizeCpu_Frame425", "d8b3f00e46a869fc"},
    {"RenderGolden_ResizeCpu_Frame426", "1d374828c2ec74d3"},
    {"RenderGolden_ResizeCpu_Frame427", "9b932c2ca2a6e315"},
    {"RenderGolden_ResizeCpu_Frame428", "c41cf3e5d225b7d5"},
    {"RenderGolden_ResizeCpu_Frame429", "88c050628f96b818"},
    {"RenderGolden_ResizeCpu_Frame43", "a7bfa37c3d93a08c"},
    {"RenderGolden_ResizeCpu_Frame430", "58aa627224e06c17"},
    {"RenderGolden_ResizeCpu_Frame431", "0e91e69a37d9804f"},
    {"RenderGolden_ResizeCpu_Frame432", "3d09fc5ec8746122"},
    {"RenderGolden_ResizeCpu_Frame433", "19bf41fbe9533481"},
    {"RenderGolden_ResizeCpu_Frame434", "5ed708dc8c4e53f4"},
    {"RenderGolden_ResizeCpu_Frame435", "8665bbea280cf19c"},
    {"RenderGolden_ResizeCpu_Frame436", "4623617dfb25e43d"},
    {"RenderGolden_ResizeCpu_Frame437", "3e4149354fa41e03"},
    {"RenderGolden_ResizeCpu_Frame438", "aacb2ca961679d94"},
    {"RenderGolden_ResizeCpu_Frame439", "d85abb0ffb2bb43a"},
    {"RenderGolden_ResizeCpu_Frame44", "a1132a3c9b5a9612"},
    {"RenderGolden_ResizeCpu_Frame440", "d00596d04ecc6005"},
    {"RenderGolden_ResizeCpu_Frame441", "fcef916c28f17300"},
    {"RenderGolden_ResizeCpu_Frame442", "67e083e8b5cf8022"},
    {"RenderGolden_ResizeCpu_Frame443", "9a4c5724cfae62da"},
    {"RenderGolden_ResizeCpu_Frame444", "bfd5a71e3ea0f7a3"},
    {"RenderGolden_ResizeCpu_Frame445", "dc28e4db82e1eed0"},
    {"RenderGolden_ResizeCpu_Frame446", "b51f08861e4b8687"},
    {"RenderGolden_ResizeCpu_Frame447", "635ea5a0035d6157"},
    {"RenderGolden_ResizeCpu_Frame448", "a349782d17189a8b"},
    {"RenderGolden_ResizeCpu_Frame449", "6d508a97431c6af5"},
    {"RenderGolden_ResizeCpu_Frame45", "6c8a5c0e63fd0771"},
    {"RenderGolden_ResizeCpu_Frame450", "ba527fb784f83f1d"},
    {"RenderGolden_ResizeCpu_Frame451", "d7cf9668cc5e19e4"},
    {"RenderGolden_ResizeCpu_Frame452", "d208fe7d994687b6"},
    {"RenderGolden_ResizeCpu_Frame453", "5dc2a23beabdd4d3"},
    {"RenderGolden_ResizeCpu_Frame454", "9bb8ade02656977c"},
    {"RenderGolden_ResizeCpu_Frame455", "ad56df8c0e394203"},
    {"RenderGolden_ResizeCpu_Frame456", "7e577be4670f006c"},
    {"RenderGolden_ResizeCpu_Frame457", "e23f1f38a59399b2"},
    {"RenderGolden_ResizeCpu_Frame458", "7d4b3a3195fd6f42"},
    {"RenderGolden_ResizeCpu_Frame459", "17c6dcae7f12c037"},
    {"RenderGolden_ResizeCpu_Frame46", "c2aaf51a85da2532"},
    {"RenderGolden_ResizeCpu_Frame460", "29aa99918080ff50"},
    {"RenderGolden_ResizeCpu_Frame461", "50a13ffebf5f3d69"},
    {"RenderGolden_ResizeCpu_Frame462", "7ce2535391b826f5"},
    {"RenderGolden_ResizeCpu_Frame463", "8c133f92d7527f6a"},
    {"RenderGolden_ResizeCpu_Frame464", "b52fbd63354a4bb9"},
    {"RenderGolden_ResizeCpu_Frame465", "33dbd760232adc79"},
    {"RenderGolden_ResizeCpu_Frame466", "f08b606c5829af53"},
    {"RenderGolden_ResizeCpu_Frame467", "bfb36cfe6ea3d016"},
    {"RenderGolden_ResizeCpu_Frame468", "e20ac41511624114"},
    {"RenderGolden_ResizeCpu_Frame469", "941d4e1d20e8b88e"},
    {"RenderGolden_ResizeCpu_Frame47", "bd0e827c9f374df4"},
    {"RenderGolden_ResizeCpu_Frame470", "97e5591102ce824a"},
    {"RenderGolden_ResizeCpu_Frame471", "2dcd5927a03fac81"},
    {"RenderGolden_ResizeCpu_Frame472", "149c245c6424b08c"},
    {"RenderGolden_ResizeCpu_Frame473", "1b6b89c8b4abc2a7"},
    {"RenderGolden_ResizeCpu_Frame474", "2460bff6009d03d4"},
    {"RenderGolden_ResizeCpu_Frame475", "4c31bf9fe532b79a"},
    {"RenderGolden_ResizeCpu_Frame476", "a2a95d5e1f69c93a"},
    {"RenderGolden_ResizeCpu_Frame477", "a6b7906c7ff3fff9"},
    {"RenderGolden_ResizeCpu_Frame478", "8ce659116cf0bc24"},
    {"RenderGolden_ResizeCpu_Frame479", "cb938486c90c3b41"},
    {"RenderGolden_ResizeCpu_Frame48", "02be70934b25351c"},
    {"RenderGolden_ResizeCpu_Frame480", "8c014a988acbd8e9"},
    {"RenderGolden_ResizeCpu_Frame481", "79e71bdc5d8a1f11"},
    {"RenderGolden_ResizeCpu_Frame482", "754c2e684f74aaea"},
    {"RenderGolden_ResizeCpu_Frame483", "561ac41e6e1efc39"},
    {"RenderGolden_ResizeCpu_Frame484", "c57cbe79af90433d"},
    {"RenderGolden_ResizeCpu_Frame485", "4578bd728294df04"},
    {"RenderGolden_ResizeCpu_Frame486", "305670286084bc20"},
    {"RenderGolden_ResizeCpu_Frame487", "234c64be83485aee"},
    {"RenderGolden_ResizeCpu_Frame488", "9351baa831589f42"},
    {"RenderGolden_ResizeCpu_Frame489", "b9f265820610363c"},
    {"RenderGolden_ResizeCpu_Frame49", "f06c5ae6a9176670"},
    {"RenderGolden_ResizeCpu_Frame490", "33e0e4fca83b6189"},
    {"RenderGolden_ResizeCpu_Frame491", "7427258865a76000"},
    {"RenderGolden_ResizeCpu_Frame492", "d7f1c4d55190fa50"},
    {"RenderGolden_ResizeCpu_Frame493", "cb2fa481cc456236"},
    {"RenderGolden_ResizeCpu_Frame494", "f8ee1d5e38354b26"},
    {"RenderGolden_ResizeCpu_Frame495", "bde52e9268a179c4"},
    {"RenderGolden_ResizeCpu_Frame496", "22dd29cd0d5043bd"},
    {"RenderGolden_ResizeCpu_Frame497", "9fcfedc8c7e96ffb"},
    {"RenderGolden_ResizeCpu_Frame498", "d3fb60f88f29a32e"},
    {"RenderGolden_ResizeCpu_Frame499", "e7bd07d17dc9053c"},
    {"RenderGolden_ResizeCpu_Frame5", "f48daf2a8fc1381a"},
    {"RenderGolden_ResizeCpu_Frame50", "941b863508afff57"},
    {"RenderGolden_ResizeCpu_Frame500", "cabbcd5a81ac242c"},
    {"RenderGolden_ResizeCpu_Frame501", "5f3f745128e5d3bc"},
    {"RenderGolden_ResizeCpu_Frame502", "a24e51615a1d6c9e"},
    {"RenderGolden_ResizeCpu_Frame503", "07475295323afdae"},
    {"RenderGolden_ResizeCpu_Frame504", "b2981d96e8f83c83"},
    {"RenderGolden_ResizeCpu_Frame505", "d7017ccb40580ad5"},
    {"RenderGolden_ResizeCpu_Frame506", "76430cbf1f0389b3"},
    {"RenderGolden_ResizeCpu_Frame507", "f8d8a13edb39cb08"},
    {"RenderGolden_ResizeCpu_Frame508", "e3d8ece3a7090d4c"},
    {"RenderGolden_ResizeCpu_Frame509", "2d127606fddf3584"},
    {"RenderGolden_ResizeCpu_Frame51", "fcc75c480caff1cc"},
    {"RenderGolden_ResizeCpu_Frame510", "7741266aaa65b297"},
    {"RenderGolden_ResizeCpu_Frame511", "d1dd97da8f8b2f25"},
    {"RenderGolden_ResizeCpu_Frame512", "9cf4ef2c3b7644f9"},
    {"RenderGolden_ResizeCpu_Frame513", "0a28d9045768bb77"},
    {"RenderGolden_ResizeCpu_Frame514", "618a52195e00ac60"},
    {"RenderGolden_ResizeCpu_Frame515", "7074a953e0761fae"},
    {"RenderGolden_ResizeCpu_Frame516", "da6c340d1b01b5b8"},
    {"RenderGolden_ResizeCpu_Frame517", "39ba8c23ad58fc41"},
    {"RenderGolden_ResizeCpu_Frame518", "e89939dc13e23722"},
    {"RenderGolden_ResizeCpu_Frame519", "850fc580d43a51c7"},
    {"RenderGolden_ResizeCpu_Frame52", "c7baab28efd0e353"},
    {"RenderGolden_ResizeCpu_Frame520", "eda87e42a5b875b2"},
    {"RenderGolden_ResizeCpu_Frame53", "499bc03501d6f86a"},
    {"RenderGolden_ResizeCpu_Frame54", "f12723d21f216378"},
    {"RenderGolden_ResizeCpu_Frame55", "34f680a0013d463f"},
    {"RenderGolden_ResizeCpu_Frame56", "c8e4db6e36bc7c0c"},
    {"RenderGolden_ResizeCpu_Frame57", "88de1203dc2120e1"},
    {"RenderGolden_ResizeCpu_Frame58", "1df55e681d65b697"},
    {"RenderGolden_ResizeCpu_Frame59", "4396b7d75d5b2b43"},
    {"RenderGolden_ResizeCpu_Frame6", "e6ece29733cd88fa"},
    {"RenderGolden_ResizeCpu_Frame60", "137ef93cfe857718"},
    {"RenderGolden_ResizeCpu_Frame61", "eed19a6f7b911149"},
    {"RenderGolden_ResizeCpu_Frame62", "3a6bedb3f4f56af3"},
    {"RenderGolden_ResizeCpu_Frame63", "8db1fe4c2abaa1d0"},
    {"RenderGolden_ResizeCpu_Frame64", "8d90736d16c75989"},
    {"RenderGolden_ResizeCpu_Frame65", "ea611c5f4eb5807a"},
    {"RenderGolden_ResizeCpu_Frame66", "3d8277231077387a"},
    {"RenderGolden_ResizeCpu_Frame67", "ea233f4c7a234eab"},
    {"RenderGolden_ResizeCpu_Frame68", "0bdfe43316199200"},
    {"RenderGolden_ResizeCpu_Frame69", "553b4de6b3add7cb"},
    {"RenderGolden_ResizeCpu_Frame7", "2da1bfafa150a28a"},
    {"RenderGolden_ResizeCpu_Frame70", "2f70c264982ff1ff"},
    {"RenderGolden_ResizeCpu_Frame71", "4ca1f95f307db9c4"},
    {"RenderGolden_ResizeCpu_Frame72", "583cf2d0ea1a4c4a"},
    {"RenderGolden_ResizeCpu_Frame73", "4eb60825f2cb158a"},
    {"RenderGolden_ResizeCpu_Frame74", "80bfc621c485b496"},
    {"RenderGolden_ResizeCpu_Frame75", "626035d792e6cce4"},
    {"RenderGolden_ResizeCpu_Frame76", "7dd2e2a06ae48657"},
    {"RenderGolden_ResizeCpu_Frame77", "b03b729c8ec26ef3"},
    {"RenderGolden_ResizeCpu_Frame78", "faad1940cf2ed71c"},
    {"RenderGolden_ResizeCpu_Frame79", "e100f84db63d5454"},
    {"RenderGolden_ResizeCpu_Frame8", "1b689dcb439d84fe"},
    {"RenderGolden_ResizeCpu_Frame80", "7a45ec46146a742d"},
    {"RenderGolden_ResizeCpu_Frame81", "339f84747b3a7c6e"},
    {"RenderGolden_ResizeCpu_Frame82", "8cec866d68bb2e58"},
    {"RenderGolden_ResizeCpu_Frame83", "163316b94878dbcd"},
    {"RenderGolden_ResizeCpu_Frame84", "315414f155678c86"},
    {"RenderGolden_ResizeCpu_Frame85", "87b0ce462f636b95"},
    {"RenderGolden_ResizeCpu_Frame86", "1162dbadede15971"},
    {"RenderGolden_ResizeCpu_Frame87", "338a03eee4a65eb5"},
    {"RenderGolden_ResizeCpu_Frame88", "a65214479fd77b9c"},
    {"RenderGolden_ResizeCpu_Frame89", "434b43bf82475c28"},
    {"RenderGolden_ResizeCpu_Frame9", "154c3ad751e23556"},
    {"RenderGolden_ResizeCpu_Frame90", "3d126b2cd483ea92"},
    {"RenderGolden_ResizeCpu_Frame91", "07339b16542dbf9f"},
    {"RenderGolden_ResizeCpu_Frame92", "0e7b88246be388ba"},
    {"RenderGolden_ResizeCpu_Frame93", "9bf141379c70cce8"},
    {"RenderGolden_ResizeCpu_Frame94", "f5e486dd529911ba"},
    {"RenderGolden_ResizeCpu_Frame95", "4a9b69a7647adc78"},
    {"RenderGolden_ResizeCpu_Frame96", "59fb43f24aedc561"},
    {"RenderGolden_ResizeCpu_Frame97", "edb699812b094308"},
    {"RenderGolden_ResizeCpu_Frame98", "880b7f85ae9a5f94"},
    {"RenderGolden_ResizeCpu_Frame99", "fd2663cbd6543cf5"},
    {"RenderGolden_ResizeGpu_Frame0", "fe644b4f14a0ee15"},
    {"RenderGolden_ResizeGpu_Frame1", "dade68c14e007db1"},
    {"RenderGolden_ResizeGpu_Frame10", "37c1951a73ab340f"},
    {"RenderGolden_ResizeGpu_Frame100", "a9cd26d0f81a217f"},
    {"RenderGolden_ResizeGpu_Frame101", "b906f82aa1a38ddb"},
    {"RenderGolden_ResizeGpu_Frame102", "40b85c1bfd30ffba"},
    {"RenderGolden_ResizeGpu_Frame103", "8123814b6f6ed262"},
    {"RenderGolden_ResizeGpu_Frame104", "02222364d66bc173"},
    {"RenderGolden_ResizeGpu_Frame105", "7dbdc18826f7f946"},
    {"RenderGolden_ResizeGpu_Frame106", "3a81af67e199b360"},
    {"RenderGolden_ResizeGpu_Frame107", "81d4035e9dc79fbc"},
    {"RenderGolden_ResizeGpu_Frame108", "fa907a13e419b3a5"},
    {"RenderGolden_ResizeGpu_Frame109", "4bba0dc253990d37"},
    {"RenderGolden_ResizeGpu_Frame11", "a67c6021064715d3"},
    {"RenderGolden_ResizeGpu_Frame110", "acd74846f5824b31"},
    {"RenderGolden_ResizeGpu_Frame111", "53eac23d941561f1"},
    {"RenderGolden_ResizeGpu_Frame112", "e7cffa16dbf18d2e"},
    {"RenderGolden_ResizeGpu_Frame113", "968f627355b69cce"},
    {"RenderGolden_ResizeGpu_Frame114", "4ea94fa8cb33bd55"},
    {"RenderGolden_ResizeGpu_Frame115", "7f525e6d2bf4b9dd"},
    {"RenderGolden_ResizeGpu_Frame116", "543e0fd4d34b64b3"},
    {"RenderGolden_ResizeGpu_Frame117", "589c87da45f7916a"},
    {"RenderGolden_ResizeGpu_Frame118", "d5510efde69e5d0e"},
    {"RenderGolden_ResizeGpu_Frame119", "573ad4134ab8be08"},
    {"RenderGolden_ResizeGpu_Frame12", "f7b5a24a61365343"},
    {"RenderGolden_ResizeGpu_Frame120", "0a2e2ada13abe3d8"},
    {"RenderGolden_ResizeGpu_Frame121", "95df566c0a2df43a"},
    {"RenderGolden_ResizeGpu_Frame122", "f497f32a134884fa"},
    {"RenderGolden_ResizeGpu_Frame123", "7a7bb27b0ed0762a"},
    {"RenderGolden_ResizeGpu_Frame124", "bef7dd20e66c0eb1"},
    {"RenderGolden_ResizeGpu_Frame125", "039d93e6fe6b5e63"},
    {"RenderGolden_ResizeGpu_Frame126", "25b7b8a17fc1f410"},
    {"RenderGolden_ResizeGpu_Frame127", "c137387352afa3d2"},
    {"RenderGolden_ResizeGpu_Frame128", "458c89ab0a87668e"},
    {"RenderGolden_ResizeGpu_Frame129", "bab3b573ea75da83"},
    {"RenderGolden_ResizeGpu_Frame13", "dca851ea815ee9b8"},
    {"RenderGolden_ResizeGpu_Frame130", "d9198e6186f12a49"},
    {"RenderGolden_ResizeGpu_Frame131", "c24caf5eb5dcd90d"},
    {"RenderGolden_ResizeGpu_Frame132", "dd0d32675b60f977"},
    {"RenderGolden_ResizeGpu_Frame133", "dd5c9b1334f30dec"},
    {"RenderGolden_ResizeGpu_Frame134", "fb3b4518fdab1207"},
    {"RenderGolden_ResizeGpu_Frame135", "b655ad142aeea2ae"},
    {"RenderGolden_ResizeGpu_Frame136", "37c30da5ca7bf1fc"},
    {"RenderGolden_ResizeGpu_Frame137", "5f9e4ff3259e916a"},
    {"RenderGolden_ResizeGpu_Frame138", "ce4332aebef20c76"},
    {"RenderGolden_ResizeGpu_Frame139", "38766e44d4efbba7"},
    {"RenderGolden_ResizeGpu_Frame14", "7720d05728e2ff67"},
    {"RenderGolden_ResizeGpu_Frame140", "b7f5ca29cc84d6f7"},
    {"RenderGolden_ResizeGpu_Frame141", "a81fccf91603de00"},
    {"RenderGolden_ResizeGpu_Frame142", "56292f7af57c9fad"},
    {"RenderGolden_ResizeGpu_Frame143", "0c0dba022b746101"},
    {"RenderGolden_ResizeGpu_Frame144", "c5f95de3fa595782"},
    {"RenderGolden_ResizeGpu_Frame145", "c9190c6714d2672e"},
    {"RenderGolden_ResizeGpu_Frame146", "7f2c16f35e5ea0a1"},
    {"RenderGolden_ResizeGpu_Frame147", "2f7cbc7496c57ac0"},
    {"RenderGolden_ResizeGpu_Frame148", "e72c38880308b7d8"},
    {"RenderGolden_ResizeGpu_Frame149", "9abf5a2140493be9"},
    {"RenderGolden_ResizeGpu_Frame15", "bd598f1fa5a62dcb"},
    {"RenderGolden_ResizeGpu_Frame150", "ca063eb2854fa58d"},
    {"RenderGolden_ResizeGpu_Frame151", "21611633aa2b2cb2"},
    {"RenderGolden_ResizeGpu_Frame152", "0368310fd072c846"},
    {"RenderGolden_ResizeGpu_Frame153", "8146453c447aee5d"},
    {"RenderGolden_ResizeGpu_Frame154", "015ac8161b59b7ca"},
    {"RenderGolden_ResizeGpu_Frame155", "d353b5da40195334"},
    {"RenderGolden_ResizeGpu_Frame156", "966d7e39e5969f3c"},
    {"RenderGolden_ResizeGpu_Frame157", "8e8716121c09a308"},
    {"RenderGolden_ResizeGpu_Frame158", "14e4639e402dfabb"},
    {"RenderGolden_ResizeGpu_Frame159", "bb5ced9af425ebf3"},
    {"RenderGolden_ResizeGpu_Frame16", "5d563459efe851e5"},
    {"RenderGolden_ResizeGpu_Frame160", "4e620641d70665ce"},
    {"RenderGolden_ResizeGpu_Frame161", "6f0ec479c6f7c4d8"},
    {"RenderGolden_ResizeGpu_Frame162", "be3be39bc1fa5618"},
    {"RenderGolden_ResizeGpu_Frame163", "afb554e618c77ad3"},
    {"RenderGolden_ResizeGpu_Frame164", "0f611992280d24c9"},
    {"RenderGolden_ResizeGpu_Frame165", "0a993a972f295d7a"},
    {"RenderGolden_ResizeGpu_Frame166", "13c285fd3420a8f2"},
    {"RenderGolden_ResizeGpu_Frame167", "2745af19d47534dd"},
    {"RenderGolden_ResizeGpu_Frame168", "29ca5c8a479f94d2"},
    {"RenderGolden_ResizeGpu_Frame169", "89965b343e83c1ba"},
    {"RenderGolden_ResizeGpu_Frame17", "5eda0bfa0e3ca63e"},
    {"RenderGolden_ResizeGpu_Frame170", "9ea5bc9353034b65"},
    {"RenderGolden_ResizeGpu_Frame171", "94a9808e894bd5c2"},
    {"RenderGolden_ResizeGpu_Frame172", "9e379243b989f7c2"},
    {"RenderGolden_ResizeGpu_Frame173", "4461fceb66e37831"},
    {"RenderGolden_ResizeGpu_Frame174", "3f4668d263e55058"},
    {"RenderGolden_ResizeGpu_Frame175", "57fd75a810f3e590"},
    {"RenderGolden_ResizeGpu_Frame176", "a4c960f30df72ca7"},
    {"RenderGolden_ResizeGpu_Frame177", "7fdae50349e653bf"},
    {"RenderGolden_ResizeGpu_Frame178", "18bda526dd7796e7"},
    {"RenderGolden_ResizeGpu_Frame179", "d4acc9e6186bd620"},
    {"RenderGolden_ResizeGpu_Frame18", "ab24df50669711c3"},
    {"RenderGolden_ResizeGpu_Frame180", "52baaea26e51a9c6"},
    {"RenderGolden_ResizeGpu_Frame181", "255a5cf14b95d776"},
    {"RenderGolden_ResizeGpu_Frame182", "9fd79620bba92fc1"},
    {"RenderGolden_ResizeGpu_Frame183", "a03bfa134e998014"},
    {"RenderGolden_ResizeGpu_Frame184", "b2a6fdc106ab1df2"},
    {"RenderGolden_ResizeGpu_Frame185", "46ee457924c47b8b"},
    {"RenderGolden_ResizeGpu_Frame186", "8b94871f2985b1c6"},
    {"RenderGolden_ResizeGpu_Frame187", "0bc49eca45d406dd"},
    {"RenderGolden_ResizeGpu_Frame188", "fc014a21149c5fb0"},
    {"RenderGolden_ResizeGpu_Frame189", "1a88748dd0215344"},
    {"RenderGolden_ResizeGpu_Frame19", "8ece594c48161d4f"},
    {"RenderGolden_ResizeGpu_Frame190", "b1f3871c2b0f1aa9"},
    {"RenderGolden_ResizeGpu_Frame191", "56c1f55dcf083836"},
    {"RenderGolden_ResizeGpu_Frame192", "73240349f178a2b5"},
    {"RenderGolden_ResizeGpu_Frame193", "e5746b46440c9f71"},
    {"RenderGolden_ResizeGpu_Frame194", "a0b161e2309c10a7"},
    {"RenderGolden_ResizeGpu_Frame195", "abb779467a2553f5"},
    {"RenderGolden_ResizeGpu_Frame196", "35bdab5292ad9966"},
    {"RenderGolden_ResizeGpu_Frame197", "6fe678b7135d0193"},
    {"RenderGolden_ResizeGpu_Frame198", "601105691e3afb90"},
    {"RenderGolden_ResizeGpu_Frame199", "ccc8c18ad97408a8"},
    {"RenderGolden_ResizeGpu_Frame2", "04dbacad4e65f64d"},
    {"RenderGolden_ResizeGpu_Frame20", "5fc8710ee9d90879"},
    {"RenderGolden_ResizeGpu_Frame200", "ba7556724ebb8f68"},
    {"RenderGolden_ResizeGpu_Frame201", "97a45dea7265671e"},
    {"RenderGolden_ResizeGpu_Frame202", "5a5b23f73812624c"},
    {"RenderGolden_ResizeGpu_Frame203", "491604c1cd195cb6"},
    {"RenderGolden_ResizeGpu_Frame204", "5af12d798b79c3f2"},
    {"RenderGolden_ResizeGpu_Frame205", "ee7edfbe77589b87"},
    {"RenderGolden_ResizeGpu_Frame206", "db3e460b9c43a115"},
    {"RenderGolden_ResizeGpu_Frame207", "3cc505aa2b1cfcc5"},
    {"RenderGolden_ResizeGpu_Frame208", "78159f9f302c0de1"},
    {"RenderGolden_ResizeGpu_Frame209", "0665f6f8480c557b"},
    {"RenderGolden_ResizeGpu_Frame21", "7c6d2fbca62831e1"},
    {"RenderGolden_ResizeGpu_Frame210", "fb8baab57f782946"},
    {"RenderGolden_ResizeGpu_Frame211", "8cbd54bf6e68ac8a"},
    {"RenderGolden_ResizeGpu_Frame212", "0fda2944d34bcdd1"},
    {"RenderGolden_ResizeGpu_Frame213", "f1223ad54d2568c0"},
    {"RenderGolden_ResizeGpu_Frame214", "36a80ad9f4e640cb"},
    {"RenderGolden_ResizeGpu_Frame215", "8da9df80d76bf247"},
    {"RenderGolden_ResizeGpu_Frame216", "2fdaf64c30de27db"},
    {"RenderGolden_ResizeGpu_Frame217", "744be79129a1085a"},
    {"RenderGolden_ResizeGpu_Frame218", "2edb6f06dec25bba"},
    {"RenderGolden_ResizeGpu_Frame219", "4f02acfefe9c852e"},
    {"RenderGolden_ResizeGpu_Frame22", "4ec473b1de795d5b"},
    {"RenderGolden_ResizeGpu_Frame220", "003a5979b1f46fec"},
    {"RenderGolden_ResizeGpu_Frame221", "b8386f8fe4a1b817"},
    {"RenderGolden_ResizeGpu_Frame222", "2e6b22f995c61a39"},
    {"RenderGolden_ResizeGpu_Frame223", "099fe2a794b1015b"},
    {"RenderGolden_ResizeGpu_Frame224", "5e3ecbbc5a125b91"},
    {"RenderGolden_ResizeGpu_Frame225", "648163029e232fc9"},
    {"RenderGolden_ResizeGpu_Frame226", "c797ad206fcb080a"},
    {"RenderGolden_ResizeGpu_Frame227", "9373993afde6cc30"},
    {"RenderGolden_ResizeGpu_Frame228", "450570669c951cda"},
    {"RenderGolden_ResizeGpu_Frame229", "df2d3e5958e99b5a"},
    {"RenderGolden_ResizeGpu_Frame23", "dcb95bba67605c69"},
    {"RenderGolden_ResizeGpu_Frame230", "86b8e6608a59f469"},
    {"RenderGolden_ResizeGpu_Frame231", "19f0ac040cd54264"},
    {"RenderGolden_ResizeGpu_Frame232", "45c936abe7bf3791"},
    {"RenderGolden_ResizeGpu_Frame233", "a8a23dc267c5f9a0"},
    {"RenderGolden_ResizeGpu_Frame234", "ece6d572717cdc20"},
    {"RenderGolden_ResizeGpu_Frame235", "f45ef9d366d75d81"},
    {"RenderGolden_ResizeGpu_Frame236", "5e41efe697b83a66"},
    {"RenderGolden_ResizeGpu_Frame237", "2efb735daa5a5c9c"},
    {"RenderGolden_ResizeGpu_Frame238", "df364eb2e00a1f4a"},
    {"RenderGolden_ResizeGpu_Frame239", "6957c838dcf06f89"},
    {"RenderGolden_ResizeGpu_Frame24", "67b36bc333f04538"},
    {"RenderGolden_ResizeGpu_Frame240", "f8c02ede6353f27e"},
    {"RenderGolden_ResizeGpu_Frame241", "d1f9cf3cd62565f6"},
    {"RenderGolden_ResizeGpu_Frame242", "14081cc3e3034f9d"},
    {"RenderGolden_ResizeGpu_Frame243", "d805dd77cd488365"},
    {"RenderGolden_ResizeGpu_Frame244", "042cbe8e41f44d2e"},
    {"RenderGolden_ResizeGpu_Frame245", "3de250d4f486d554"},
    {"RenderGolden_ResizeGpu_Frame246", "e8f052edb8518658"},
    {"RenderGolden_ResizeGpu_Frame247", "aa2667136e370a01"},
    {"RenderGolden_ResizeGpu_Frame248", "2a7ce6c9842c9c0c"},
    {"RenderGolden_ResizeGpu_Frame249", "75941f2093ec2444"},
    {"RenderGolden_ResizeGpu_Frame25", "bc692d8103da82e0"},
    {"RenderGolden_ResizeGpu_Frame250", "0da18fff64be58b4"},
    {"RenderGolden_ResizeGpu_Frame251", "a267a30b63789bf1"},
    {"RenderGolden_ResizeGpu_Frame252", "c8a1ef752a259d2a"},
    {"RenderGolden_ResizeGpu_Frame253", "28f54674abac50e4"},
    {"RenderGolden_ResizeGpu_Frame254", "e086caca5fd1aaf4"},
    {"RenderGolden_ResizeGpu_Frame255", "050889b1827cb57d"},
    {"RenderGolden_ResizeGpu_Frame256", "dfd54c03c575e69a"},
    {"RenderGolden_ResizeGpu_Frame257", "7b2a77c83cf2ee9f"},
    {"RenderGolden_ResizeGpu_Frame258", "3bc60d311a556bd1"},
    {"RenderGolden_ResizeGpu_Frame259", "02a0540eb459347e"},
    {"RenderGolden_ResizeGpu_Frame26", "9a0aba71f0efd382"},
    {"RenderGolden_ResizeGpu_Frame260", "7ea19731afe054d5"},
    {"RenderGolden_ResizeGpu_Frame261", "78007343b9a4971d"},
    {"RenderGolden_ResizeGpu_Frame262", "a2cc513014710057"},
    {"RenderGolden_ResizeGpu_Frame263", "1af2530282cccf83"},
    {"RenderGolden_ResizeGpu_Frame264", "c696a56f385f01c6"},
    {"RenderGolden_ResizeGpu_Frame265", "9f65c249862c4c07"},
    {"RenderGolden_ResizeGpu_Frame266", "075b35868be5a750"},
    {"RenderGolden_ResizeGpu_Frame267", "5c5a194687e83f12"},
    {"RenderGolden_ResizeGpu_Frame268", "320b5b5dbf1aa886"},
    {"RenderGolden_ResizeGpu_Frame269", "ccaa3e505e2b65ce"},
    {"RenderGolden_ResizeGpu_Frame27", "ad0b195726b8967a"},
    {"RenderGolden_ResizeGpu_Frame270", "9bc3ed0a822893d2"},
    {"RenderGolden_ResizeGpu_Frame271", "3b153c5405d8034a"},
    {"RenderGolden_ResizeGpu_Frame272", "dee954faed241d78"},
    {"RenderGolden_ResizeGpu_Frame273", "736455f01cf4bb33"},
    {"RenderGolden_ResizeGpu_Frame274", "332684e4a14ff7e0"},
    {"RenderGolden_ResizeGpu_Frame275", "9c26b24b8e30099c"},
    {"RenderGolden_ResizeGpu_Frame276", "63830de0e7e52b4a"},
    {"RenderGolden_ResizeGpu_Frame277", "a5c42d18a6d2d439"},
    {"RenderGolden_ResizeGpu_Frame278", "74b3ce597f416f4b"},
    {"RenderGolden_ResizeGpu_Frame279", "f7ec4d863fb7e8c4"},
    {"RenderGolden_ResizeGpu_Frame28", "a535991f1852ce33"},
    {"RenderGolden_ResizeGpu_Frame280", "495d6f17680b1d82"},
    {"RenderGolden_ResizeGpu_Frame281", "a7ed5d884c34fb89"},
    {"RenderGolden_ResizeGpu_Frame282", "728b58a947bac34e"},
    {"RenderGolden_ResizeGpu_Frame283", "1c5adf0aeb704f33"},
    {"RenderGolden_ResizeGpu_Frame284", "a3f01bd0a54ee793"},
    {"RenderGolden_ResizeGpu_Frame285", "ba1ad7a7f4b77693"},
    {"RenderGolden_ResizeGpu_Frame286", "bc54f98d8f27b3b5"},
    {"RenderGolden_ResizeGpu_Frame287", "17f96ed035652c8b"},
    {"RenderGolden_ResizeGpu_Frame288", "07f584c2afd7421a"},
    {"RenderGolden_ResizeGpu_Frame289", "b4c89da39599a81a"},
    {"RenderGolden_ResizeGpu_Frame29", "0ff3b321137c2239"},
    {"RenderGolden_ResizeGpu_Frame290", "e7209569b508a182"},
    {"RenderGolden_ResizeGpu_Frame291", "ad7623ef27ec43ec"},
    {"RenderGolden_ResizeGpu_Frame292", "db941200ad558e3b"},
    {"RenderGolden_ResizeGpu_Frame293", "c0eaf83e8faf6122"},
    {"RenderGolden_ResizeGpu_Frame294", "455cab5911944ee9"},
    {"RenderGolden_ResizeGpu_Frame295", "f7d2d47bcb249894"},
    {"RenderGolden_ResizeGpu_Frame296", "851c9cbc5e64c56f"},
    {"RenderGolden_ResizeGpu_Frame297", "9b92551ae11c5bad"},
    {"RenderGolden_ResizeGpu_Frame298", "2afabedf081171db"},
    {"RenderGolden_ResizeGpu_Frame299", "f0fb98faf1a86ee5"},
    {"RenderGolden_ResizeGpu_Frame3", "0cbe8fc954054a5d"},
    {"RenderGolden_ResizeGpu_Frame30", "7108ffef75d5a44f"},
    {"RenderGolden_ResizeGpu_Frame300", "d2d6df6f1b1bf203"},
    {"RenderGolden_ResizeGpu_Frame301", "14c0e45f198e7226"},
    {"RenderGolden_ResizeGpu_Frame302", "b190355589a85544"},
    {"RenderGolden_ResizeGpu_Frame303", "db94af08ab56c14a"},
    {"RenderGolden_ResizeGpu_Frame304", "c911787ac6aa4efc"},
    {"RenderGolden_ResizeGpu_Frame305", "3af8ca5897358438"},
    {"RenderGolden_ResizeGpu_Frame306", "fc3cfca709e4a865"},
    {"RenderGolden_ResizeGpu_Frame307", "689a02e93d0a802b"},
    {"RenderGolden_ResizeGpu_Frame308", "56c199ed19732e65"},
    {"RenderGolden_ResizeGpu_Frame309", "40c7b8ddff962464"},
    {"RenderGolden_ResizeGpu_Frame31", "e4854b4ec6c2facd"},
    {"RenderGolden_ResizeGpu_Frame310", "60e82cad982c2f66"},
    {"RenderGolden_ResizeGpu_Frame311", "e80694131136207c"},
    {"RenderGolden_ResizeGpu_Frame312", "bd592a59061fc0a9"},
    {"RenderGolden_ResizeGpu_Frame313", "dcd13c18c6a0ed5a"},
    {"RenderGolden_ResizeGpu_Frame314", "2748d9e33f3ae33d"},
    {"RenderGolden_ResizeGpu_Frame315", "803e0daa61b6e9e2"},
    {"RenderGolden_ResizeGpu_Frame316", "1943d3cbfac5bbfe"},
    {"RenderGolden_ResizeGpu_Frame317", "4337fecca19674cc"},
    {"RenderGolden_ResizeGpu_Frame318", "223a988e627d8c42"},
    {"RenderGolden_ResizeGpu_Frame319", "a76bc787691fc2f3"},
    {"RenderGolden_ResizeGpu_Frame32", "e9c3543d50fed0fa"},
    {"RenderGolden_ResizeGpu_Frame320", "3ac2b971c53819d4"},
    {"RenderGolden_ResizeGpu_Frame321", "791e7b384d8a7bbd"},
    {"RenderGolden_ResizeGpu_Frame322", "2fc5946209a9d252"},
    {"RenderGolden_ResizeGpu_Frame323", "ac1ac2a18d6a763b"},
    {"RenderGolden_ResizeGpu_Frame324", "dad95899db74d3a7"},
    {"RenderGolden_ResizeGpu_Frame325", "14a200bb1e9f97b2"},
    {"RenderGolden_ResizeGpu_Frame326", "962aec3d35196975"},
    {"RenderGolden_ResizeGpu_Frame327", "b0f8dbd7f73c2211"},
    {"RenderGolden_ResizeGpu_Frame328", "dffe4626615a0020"},
    {"RenderGolden_ResizeGpu_Frame329", "468b04373dd0e622"},
    {"RenderGolden_ResizeGpu_Frame33", "e51fc25a1ff88f79"},
    {"RenderGolden_ResizeGpu_Frame330", "d884b061c71656fb"},
    {"RenderGolden_ResizeGpu_Frame331", "a0441396562799b0"},
    {"RenderGolden_ResizeGpu_Frame332", "c79c44e66d4d63cc"},
    {"RenderGolden_ResizeGpu_Frame333", "6e9d31bd72d9aadb"},
    {"RenderGolden_ResizeGpu_Frame334", "abdb7871d830cc1c"},
    {"RenderGolden_ResizeGpu_Frame335", "29d9170bd9c57e90"},
    {"RenderGolden_ResizeGpu_Frame336", "5cfaad5f8631856a"},
    {"RenderGolden_ResizeGpu_Frame337", "1c6fdde93163b01b"},
    {"RenderGolden_ResizeGpu_Frame338", "d0af60b3f00c0dbf"},
    {"RenderGolden_ResizeGpu_Frame339", "97ec13c51508eb03"},
    {"RenderGolden_ResizeGpu_Frame34", "aa53824a7986dc8a"},
    {"RenderGolden_ResizeGpu_Frame340", "e766b7849617ad82"},
    {"RenderGolden_ResizeGpu_Frame341", "f1333ddd163009de"},
    {"RenderGolden_ResizeGpu_Frame342", "8b4162c3ceda5514"},
    {"RenderGolden_ResizeGpu_Frame343", "1f659244955b713f"},
    {"RenderGolden_ResizeGpu_Frame344", "24ab6d37668a88ab"},
    {"RenderGolden_ResizeGpu_Frame345", "984da5fdf1f1442a"},
    {"RenderGolden_ResizeGpu_Frame346", "50df874bf4a21445"},
    {"RenderGolden_ResizeGpu_Frame347", "6c4fb4d1ed05833a"},
    {"RenderGolden_ResizeGpu_Frame348", "5ac985435a748e80"},
    {"RenderGolden_ResizeGpu_Frame349", "1023ffa8575de70a"},
    {"RenderGolden_ResizeGpu_Frame35", "97149a322bebf6d8"},
    {"RenderGolden_ResizeGpu_Frame350", "2a79eff104389632"},
    {"RenderGolden_ResizeGpu_Frame351", "12fde795728af174"},
    {"RenderGolden_ResizeGpu_Frame352", "949b929544a1aab8"},
    {"RenderGolden_ResizeGpu_Frame353", "8d7c054db1c2fcb2"},
    {"RenderGolden_ResizeGpu_Frame354", "c7d23b02999f347b"},
    {"RenderGolden_ResizeGpu_Frame355", "2ad64a62e7bc2cb9"},
    {"RenderGolden_ResizeGpu_Frame356", "fc4448efc91fb228"},
    {"RenderGolden_ResizeGpu_Frame357", "68d572876932ce73"},
    {"RenderGolden_ResizeGpu_Frame358", "79bc622ce884ab0a"},
    {"RenderGolden_ResizeGpu_Frame359", "7cf9da64a9876ddb"},
    {"RenderGolden_ResizeGpu_Frame36", "863b6af7c74788f7"},
    {"RenderGolden_ResizeGpu_Frame360", "fa354dc88beaca85"},
    {"RenderGolden_ResizeGpu_Frame361", "8c96fd4e838149ee"},
    {"RenderGolden_ResizeGpu_Frame362", "762658710e326a32"},
    {"RenderGolden_ResizeGpu_Frame363", "71b6ade6b17c413d"},
    {"RenderGolden_ResizeGpu_Frame364", "acde8dcf8002abf4"},
    {"RenderGolden_ResizeGpu_Frame365", "9aa84e7e52013d38"},
    {"RenderGolden_ResizeGpu_Frame366", "f239462d356872a3"},
    {"RenderGolden_ResizeGpu_Frame367", "b7871cf719406334"},
    {"RenderGolden_ResizeGpu_Frame368", "f1a0e6aeb25326d2"},
    {"RenderGolden_ResizeGpu_Frame369", "219d2786652aa5cd"},
    {"RenderGolden_ResizeGpu_Frame37", "29d0d1337a72c71d"},
    {"RenderGolden_ResizeGpu_Frame370", "0e1436f9e72e7569"},
    {"RenderGolden_ResizeGpu_Frame371", "a6fa17326a98b36c"},
    {"RenderGolden_ResizeGpu_Frame372", "ce1007498ef2295f"},
    {"RenderGolden_ResizeGpu_Frame373", "9fb12e081064cb60"},
    {"RenderGolden_ResizeGpu_Frame374", "522eec553ba8d41c"},
    {"RenderGolden_ResizeGpu_Frame375", "3b8b8f3c9d84b29d"},
    {"RenderGolden_ResizeGpu_Frame376", "2ea1aeb8566f72ac"},
    {"RenderGolden_ResizeGpu_Frame377", "046b5c7ec29d9577"},
    {"RenderGolden_ResizeGpu_Frame378", "f2159c2c8e9db20a"},
    {"RenderGolden_ResizeGpu_Frame379", "2d2e5f4c551a44bb"},
    {"RenderGolden_ResizeGpu_Frame38", "b66e553601d98fb8"},
    {"RenderGolden_ResizeGpu_Frame380", "61a8c1c11bb689f2"},
    {"RenderGolden_ResizeGpu_Frame381", "6965749beef216d9"},
    {"RenderGolden_ResizeGpu_Frame382", "7cfb24c6bc28e520"},
    {"RenderGolden_ResizeGpu_Frame383", "385f88b9917c6919"},
    {"RenderGolden_ResizeGpu_Frame384", "d3e60ffb469c6512"},
    {"RenderGolden_ResizeGpu_Frame385", "ad9e7767b2dffa4b"},
    {"RenderGolden_ResizeGpu_Frame386", "e57edcfc730ea9c2"},
    {"RenderGolden_ResizeGpu_Frame387", "17c6ed3ff6dcae71"},
    {"RenderGolden_ResizeGpu_Frame388", "d189a99f43d6d103"},
    {"RenderGolden_ResizeGpu_Frame389", "6c522008bfb970e0"},
    {"RenderGolden_ResizeGpu_Frame39", "26f86c8fa8aae0ee"},
    {"RenderGolden_ResizeGpu_Frame390", "56601b8e3c3ab528"},
    {"RenderGolden_ResizeGpu_Frame391", "72bd5fd13482a20b"},
    {"RenderGolden_ResizeGpu_Frame392", "9847846214065a00"},
    {"RenderGolden_ResizeGpu_Frame393", "486a13d8d075920c"},
    {"RenderGolden_ResizeGpu_Frame394", "36aa0d4a3be7a337"},
    {"RenderGolden_ResizeGpu_Frame395", "cc88503009ac754b"},
    {"RenderGolden_ResizeGpu_Frame396", "00a0eafcf9102cce"},
    {"RenderGolden_ResizeGpu_Frame397", "8ea449b5800e78c3"},
    {"RenderGolden_ResizeGpu_Frame398", "9bc80673d910a47c"},
    {"RenderGolden_ResizeGpu_Frame399", "e695922e2598fe8d"},
    {"RenderGolden_ResizeGpu_Frame4", "758420d6e7ae5c01"},
    {"RenderGolden_ResizeGpu_Frame40", "5406f05e1ec87535"},
    {"RenderGolden_ResizeGpu_Frame400", "7c92f8c7957b9cb5"},
    {"RenderGolden_ResizeGpu_Frame401", "3392da975fbb1441"},
    {"RenderGolden_ResizeGpu_Frame402", "d64828f650c496f2"},
    {"RenderGolden_ResizeGpu_Frame403", "eee042a60d7c18c4"},
    {"RenderGolden_ResizeGpu_Frame404", "ce87437e4ab3a6a1"},
    {"RenderGolden_ResizeGpu_Frame405", "1c7d4f19c226e6d3"},
    {"RenderGolden_ResizeGpu_Frame406", "37c4f0a2850daee2"},
    {"RenderGolden_ResizeGpu_Frame407", "e160e5fe180f4e5f"},
    {"RenderGolden_ResizeGpu_Frame408", "fb10dc504cf9b66c"},
    {"RenderGolden_ResizeGpu_Frame409", "f5eff461bfda42db"},
    {"RenderGolden_ResizeGpu_Frame41", "df41151a97bda8f2"},
    {"RenderGolden_ResizeGpu_Frame410", "70186da8b2059394"},
    {"RenderGolden_ResizeGpu_Frame411", "2b4d8101abcd8df1"},
    {"RenderGolden_ResizeGpu_Frame412", "3ab5eef2ac93271a"},
    {"RenderGolden_ResizeGpu_Frame413", "5865cac75c4cba1e"},
    {"RenderGolden_ResizeGpu_Frame414", "ed718242ee6aece1"},
    {"RenderGolden_ResizeGpu_Frame415", "6003323eeae9ef41"},
    {"RenderGolden_ResizeGpu_Frame416", "f54deed4941a9f98"},
    {"RenderGolden_ResizeGpu_Frame417", "d37d4707be067b88"},
    {"RenderGolden_ResizeGpu_Frame418", "6470a169491e16fd"},
    {"RenderGolden_ResizeGpu_Frame419", "f668801d89ced73b"},
    {"RenderGolden_ResizeGpu_Frame42", "448964218e86fe46"},
    {"RenderGolden_ResizeGpu_Frame420", "c4ca197ed2bc25a9"},
    {"RenderGolden_ResizeGpu_Frame421", "df14ded2d1127a6b"},
    {"RenderGolden_ResizeGpu_Frame422", "e2b905bcb02c052b"},
    {"RenderGolden_ResizeGpu_Frame423", "464e685dea5569a1"},
    {"RenderGolden_ResizeGpu_Frame424", "f5cd96bdeb42d304"},
    {"RenderGolden_ResizeGpu_Frame425", "7f20455c0eaafd9e"},
    {"RenderGolden_ResizeGpu_Frame426", "f74e9f8aff2a3f0d"},
    {"RenderGolden_ResizeGpu_Frame427", "b2dc83464a40e76d"},
    {"RenderGolden_ResizeGpu_Frame428", "d446b52fd8b47c81"},
    {"RenderGolden_ResizeGpu_Frame429", "8755b80cbba4d21c"},
    {"RenderGolden_ResizeGpu_Frame43", "12cd70048dd12308"},
    {"RenderGolden_ResizeGpu_Frame430", "e3eb269c2194a483"},
    {"RenderGolden_ResizeGpu_Frame431", "bb39b0d1a80aa7b5"},
    {"RenderGolden_ResizeGpu_Frame432", "8c9af03d6a26388e"},
    {"RenderGolden_ResizeGpu_Frame433", "d89e40cfc34acc22"},
    {"RenderGolden_ResizeGpu_Frame434", "cbf67be3f83786d9"},
    {"RenderGolden_ResizeGpu_Frame435", "56e840eb91c32ba3"},
    {"RenderGolden_ResizeGpu_Frame436", "e72c29aa91a0ebc4"},
    {"RenderGolden_ResizeGpu_Frame437", "8170fa3da1c1bf42"},
    {"RenderGolden_ResizeGpu_Frame438", "c90f7d23ae253fcc"},
    {"RenderGolden_ResizeGpu_Frame439", "9010e983860a3f8c"},
    {"RenderGolden_ResizeGpu_Frame44", "fbcb4a01db8d4291"},
    {"RenderGolden_ResizeGpu_Frame440", "6d17833ded31d759"},
    {"RenderGolden_ResizeGpu_Frame441", "448440c2c8910b44"},
    {"RenderGolden_ResizeGpu_Frame442", "c4f7a51a1aa2a9ad"},
    {"RenderGolden_ResizeGpu_Frame443", "16cb38ed3e4c1086"},
    {"RenderGolden_ResizeGpu_Frame444", "6d11638229e5262e"},
    {"RenderGolden_ResizeGpu_Frame445", "57957c9806e3266b"},
    {"RenderGolden_ResizeGpu_Frame446", "3537547fcde7176d"},
    {"RenderGolden_ResizeGpu_Frame447", "852f0af467ec1750"},
    {"RenderGolden_ResizeGpu_Frame448", "a83cfcca94b40a51"},
    {"RenderGolden_ResizeGpu_Frame449", "cbb7127a073cddfd"},
    {"RenderGolden_ResizeGpu_Frame45", "c0760cd13181c0db"},
    {"RenderGolden_ResizeGpu_Frame450", "24f7883ace0743f7"},
    {"RenderGolden_ResizeGpu_Frame451", "f54ca02bcbb7f261"},
    {"RenderGolden_ResizeGpu_Frame452", "2839adb1dd23c238"},
    {"RenderGolden_ResizeGpu_Frame453", "f85f25d60c357231"},
    {"RenderGolden_ResizeGpu_Frame454", "5047d45979676b19"},
    {"RenderGolden_ResizeGpu_Frame455", "f73628cc5faf6f60"},
    {"RenderGolden_ResizeGpu_Frame456", "8fd5b48b8e39fbd4"},
    {"RenderGolden_ResizeGpu_Frame457", "a0f8c704e23fe868"},
    {"RenderGolden_ResizeGpu_Frame458", "169bb7cecc0da876"},
    {"RenderGolden_ResizeGpu_Frame459", "8ef9dde8b5a90aae"},
    {"RenderGolden_ResizeGpu_Frame46", "e606072e66bb3866"},
    {"RenderGolden_ResizeGpu_Frame460", "d800dee6cb71300d"},
    {"RenderGolden_ResizeGpu_Frame461", "b6e41ccc6fb0328d"},
    {"RenderGolden_ResizeGpu_Frame462", "786aa368fc9ec1c9"},
    {"RenderGolden_ResizeGpu_Frame463", "57b1342bf330f466"},
    {"RenderGolden_ResizeGpu_Frame464", "7e58a87df5451fe5"},
    {"RenderGolden_ResizeGpu_Frame465", "25e75de21d353ebe"},
    {"RenderGolden_ResizeGpu_Frame466", "4bc24c8e921f53df"},
    {"RenderGolden_ResizeGpu_Frame467", "9e533fc6dd800136"},
    {"RenderGolden_ResizeGpu_Frame468", "7439d5a0a2f3fb64"},
    {"RenderGolden_ResizeGpu_Frame469", "54a22969b48d7ae8"},
    {"RenderGolden_ResizeGpu_Frame47", "d38e4eeb4b702888"},
    {"RenderGolden_ResizeGpu_Frame470", "2fae84b0f67b2e6c"},
    {"RenderGolden_ResizeGpu_Frame471", "0d526e5c1edc64a6"},
    {"RenderGolden_ResizeGpu_Frame472", "4c74fcbda350fc4c"},
    {"RenderGolden_ResizeGpu_Frame473", "25d32cee2e2acf45"},
    {"RenderGolden_ResizeGpu_Frame474", "f8e96793f0fdf525"},
    {"RenderGolden_ResizeGpu_Frame475", "2ee5465d66ee2c0a"},
    {"RenderGolden_ResizeGpu_Frame476", "0d2d4b24d2a3b335"},
    {"RenderGolden_ResizeGpu_Frame477", "5a357f532dcb7db6"},
    {"RenderGolden_ResizeGpu_Frame478", "3d3dc5e436def921"},
    {"RenderGolden_ResizeGpu_Frame479", "594da7ef3c7f7a82"},
    {"RenderGolden_ResizeGpu_Frame48", "b0e243d39646a1b0"},
    {"RenderGolden_ResizeGpu_Frame480", "0ae025031cc8b0f0"},
    {"RenderGolden_ResizeGpu_Frame481", "e721b69fb27f8144"},
    {"RenderGolden_ResizeGpu_Frame482", "2861e13461d0f1c6"},
    {"RenderGolden_ResizeGpu_Frame483", "df46ea10e8f446b7"},
    {"RenderGolden_ResizeGpu_Frame484", "5893de2169052095"},
    {"RenderGolden_ResizeGpu_Frame485", "56a65d3b3498d1da"},
    {"RenderGolden_ResizeGpu_Frame486", "e7f16026a10c3ca3"},
    {"RenderGolden_ResizeGpu_Frame487", "5ac6a35e24d7b007"},
    {"RenderGolden_ResizeGpu_Frame488", "94d08bf0baa24a8e"},
    {"RenderGolden_ResizeGpu_Frame489", "0aafb23c1bb3cb63"},
    {"RenderGolden_ResizeGpu_Frame49", "6e1a99b0a2139420"},
    {"RenderGolden_ResizeGpu_Frame490", "6ac93d065c7e0d13"},
    {"RenderGolden_ResizeGpu_Frame491", "1801da8fd6f6e9f4"},
    {"RenderGolden_ResizeGpu_Frame492", "487a932d9ffc36fa"},
    {"RenderGolden_ResizeGpu_Frame493", "e77cad96d8944d0c"},
    {"RenderGolden_ResizeGpu_Frame494", "6b99b22629a36fc4"},
    {"RenderGolden_ResizeGpu_Frame495", "6eeeb81fcc1a9aa1"},
    {"RenderGolden_ResizeGpu_Frame496", "610d64fb2e9b8a4c"},
    {"RenderGolden_ResizeGpu_Frame497", "33e523eae79891e1"},
    {"RenderGolden_ResizeGpu_Frame498", "cad27e0652cc4326"},
    {"RenderGolden_ResizeGpu_Frame499", "37d6525d76605915"},
    {"RenderGolden_ResizeGpu_Frame5", "0c72e5b6a8582883"},
    {"RenderGolden_ResizeGpu_Frame50", "bbd9de63bce65da5"},
    {"RenderGolden_ResizeGpu_Frame500", "05dff304a8ef1ecf"},
    {"RenderGolden_ResizeGpu_Frame501", "2304a05be77f512a"},
    {"RenderGolden_ResizeGpu_Frame502", "b8187d744d89e030"},
    {"RenderGolden_ResizeGpu_Frame503", "f5eaa9ec5223ad48"},
    {"RenderGolden_ResizeGpu_Frame504", "b75ce16454d012c3"},
    {"RenderGolden_ResizeGpu_Frame505", "bc5ddbbfc900479d"},
    {"RenderGolden_ResizeGpu_Frame506", "52e2c66aaf6cc86a"},
    {"RenderGolden_ResizeGpu_Frame507", "6301b411f8f8b267"},
    {"RenderGolden_ResizeGpu_Frame508", "1d2d158cf237ad5f"},
    {"RenderGolden_ResizeGpu_Frame509", "70d4b2e4357ac587"},
    {"RenderGolden_ResizeGpu_Frame51", "9304a8da797b1e2b"},
    {"RenderGolden_ResizeGpu_Frame510", "5193ae4e5e49b8cb"},
    {"RenderGolden_ResizeGpu_Frame511", "733820363d6475c4"},
    {"RenderGolden_ResizeGpu_Frame512", "2a392f52193ef9e7"},
    {"RenderGolden_ResizeGpu_Frame513", "da9c5ab2bc74bb6a"},
    {"RenderGolden_ResizeGpu_Frame514", "5f213cdbcb6b3559"},
    {"RenderGolden_ResizeGpu_Frame515", "d4e5fec8b082c705"},
    {"RenderGolden_ResizeGpu_Frame516", "337d903784aff7a7"},
    {"RenderGolden_ResizeGpu_Frame517", "e46f341a2dc374f1"},
    {"RenderGolden_ResizeGpu_Frame518", "9ec6191aeae113b6"},
    {"RenderGolden_ResizeGpu_Frame519", "440ce4bb459ba661"},
    {"RenderGolden_ResizeGpu_Frame52", "1b185bd0e50851a4"},
    {"RenderGolden_ResizeGpu_Frame520", "c90a4e8ceb1b38b6"},
    {"RenderGolden_ResizeGpu_Frame53", "597a04d5376a74ca"},
    {"RenderGolden_ResizeGpu_Frame54", "35cbf917fe5f3c5d"},
    {"RenderGolden_ResizeGpu_Frame55", "e74f9d4696a3dbb0"},
    {"RenderGolden_ResizeGpu_Frame56", "787da65132651640"},
    {"RenderGolden_ResizeGpu_Frame57", "332cf75785cd86c9"},
    {"RenderGolden_ResizeGpu_Frame58", "856b156b5668d20a"},
    {"RenderGolden_ResizeGpu_Frame59", "0d106840ff042ed0"},
    {"RenderGolden_ResizeGpu_Frame6", "c69316834ed53cb8"},
    {"RenderGolden_ResizeGpu_Frame60", "d401f519bb2bde28"},
    {"RenderGolden_ResizeGpu_Frame61", "743ca83042c31904"},
    {"RenderGolden_ResizeGpu_Frame62", "3bb7db6605620a0a"},
    {"RenderGolden_ResizeGpu_Frame63", "452e1e7986a34938"},
    {"RenderGolden_ResizeGpu_Frame64", "a646688e4c4083ea"},
    {"RenderGolden_ResizeGpu_Frame65", "56b4df752f165bba"},
    {"RenderGolden_ResizeGpu_Frame66", "d3b97d91f262b522"},
    {"RenderGolden_ResizeGpu_Frame67", "a609e6fde0b607f9"},
    {"RenderGolden_ResizeGpu_Frame68", "c09833abf6bb4f8d"},
    {"RenderGolden_ResizeGpu_Frame69", "5d9e74cae6ce4122"},
    {"RenderGolden_ResizeGpu_Frame7", "9ec14ecdb41e670e"},
    {"RenderGolden_ResizeGpu_Frame70", "4f50d9c9a239aa10"},
    {"RenderGolden_ResizeGpu_Frame71", "6e05aa474eb25c36"},
    {"RenderGolden_ResizeGpu_Frame72", "6667b3d5574dccde"},
    {"RenderGolden_ResizeGpu_Frame73", "cafad675de1177e0"},
    {"RenderGolden_ResizeGpu_Frame74", "362af6baee34fd72"},
    {"RenderGolden_ResizeGpu_Frame75", "3c31391ff6dfd768"},
    {"RenderGolden_ResizeGpu_Frame76", "2ea50bd3b712cf48"},
    {"RenderGolden_ResizeGpu_Frame77", "5be6136b5321e66c"},
    {"RenderGolden_ResizeGpu_Frame78", "d0ccf298eaca489a"},
    {"RenderGolden_ResizeGpu_Frame79", "852e65d9118a9841"},
    {"RenderGolden_ResizeGpu_Frame8", "413e5722622728af"},
    {"RenderGolden_ResizeGpu_Frame80", "71ec511fa3f8f3aa"},
    {"RenderGolden_ResizeGpu_Frame81", "23a231c81d3afc19"},
    {"RenderGolden_ResizeGpu_Frame82", "c87d678ae39f5898"},
    {"RenderGolden_ResizeGpu_Frame83", "fabfbf72b48e86c2"},
    {"RenderGolden_ResizeGpu_Frame84", "aa2d94dd33b5fe10"},
    {"RenderGolden_ResizeGpu_Frame85", "38bf83d752039cbf"},
    {"RenderGolden_ResizeGpu_Frame86", "0351a8013cd92784"},
    {"RenderGolden_ResizeGpu_Frame87", "e4b1000893735243"},
    {"RenderGolden_ResizeGpu_Frame88", "5276dcf34f928863"},
    {"RenderGolden_ResizeGpu_Frame89", "1587aab90ef17f89"},
    {"RenderGolden_ResizeGpu_Frame9", "45fd99edd4180d94"},
    {"RenderGolden_ResizeGpu_Frame90", "2fb7fbc26c3ccd6f"},
    {"RenderGolden_ResizeGpu_Frame91", "50a3d9fb652ec5b3"},
    {"RenderGolden_ResizeGpu_Frame92", "a9c4fb5f33ccacac"},
    {"RenderGolden_ResizeGpu_Frame93", "bb75c6ad1e7394a3"},
    {"RenderGolden_ResizeGpu_Frame94", "46ddd748cc188553"},
    {"RenderGolden_ResizeGpu_Frame95", "8ea022b7b0044fd7"},
    {"RenderGolden_ResizeGpu_Frame96", "d712ce8fc3977779"},
    {"RenderGolden_ResizeGpu_Frame97", "e53f47a4e492fb20"},
    {"RenderGolden_ResizeGpu_Frame98", "12bb7aa317bd3b13"},
    {"RenderGolden_ResizeGpu_Frame99", "af21a6cb773e5dce"},
    {"RenderRegression_VectorBillion", "INCOMPLETE"},
}};
inline constexpr std::string_view IncompleteCrc = "INCOMPLETE";
} // namespace RenderTests
