#include "stdafx.h"
#include "Vectors.h"

#include "GPU_Render.h"
#include "HDRFloat.h"
#include "LAInfoDeep.h"

std::wstring GetFileExtension(GrowableVectorTypes Type) {
    switch (Type) {
    case GrowableVectorTypes::Metadata:
        return L".met";
    case GrowableVectorTypes::GPUReferenceIter:
        return L".FullOrbit";
    case GrowableVectorTypes::LAStageInfo:
        return L".LAStages";
    case GrowableVectorTypes::LAInfoDeep:
        return L".LAs";
    default:
        assert(false);
        return L".error";
    }
}

template<class EltT>
EltT* GrowableVector<EltT>::GetData() const {
    return m_Data;
}

template<class EltT>
std::wstring GrowableVector<EltT>::GetFilename() const {
    return m_Filename;
}

template<class EltT>
size_t GrowableVector<EltT>::GetCapacity() const {
    return m_CapacityInElts;
}

template<class EltT>
size_t GrowableVector<EltT>::GetSize() const {
    return m_UsedSizeInElts;
}

template<class EltT>
bool GrowableVector<EltT>::ValidFile() const {
    return m_Data != nullptr;
}

template<class EltT>
GrowableVector<EltT>::GrowableVector(GrowableVector<EltT>&& other) :
    m_FileHandle{ other.m_FileHandle },
    m_MappedFile{ other.m_MappedFile },
    m_UsedSizeInElts{ other.m_UsedSizeInElts },
    m_CapacityInElts{ other.m_CapacityInElts },
    m_Data{ other.m_Data },
    m_AddPointOptions{ other.m_AddPointOptions },
    m_Filename{ other.m_Filename } {

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    other.m_Data = nullptr;
}

template<class EltT>
GrowableVector<EltT>& GrowableVector<EltT>::operator=(GrowableVector<EltT>&& other) {
    m_FileHandle = other.m_FileHandle;
    m_MappedFile = other.m_MappedFile;
    m_UsedSizeInElts = other.m_UsedSizeInElts;
    m_CapacityInElts = other.m_CapacityInElts;
    m_Data = other.m_Data;
    m_AddPointOptions = other.m_AddPointOptions;
    m_Filename = other.m_Filename;

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_Data = nullptr;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    other.m_Filename = {};

    return *this;
}

template<class EltT>
GrowableVector<EltT>::GrowableVector()
    : GrowableVector(AddPointOptions::DontSave, L"") {
    // Default to anonymous memory.
}

// The constructor takes the file to open or create
// It maps enough memory to accomodate the provided orbit size.
template<class EltT>
GrowableVector<EltT>::GrowableVector(AddPointOptions add_point_options, std::wstring filename)
    : m_FileHandle{},
    m_MappedFile{},
    m_UsedSizeInElts{},
    m_CapacityInElts{},
    m_Data{},
    m_AddPointOptions{ add_point_options },
    m_Filename{ filename }
{
    auto ret = GetPhysicallyInstalledSystemMemory(&m_PhysicalMemoryCapacityKB);
    if (ret == FALSE) {
        ::MessageBox(nullptr, L"Failed to get system memory", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        // 32 MB.  We're not allocating some function of RAM on disk.
        // It should be doing a sparse allocation but who knows.
        MutableFileCommit(1024);
    }
}

// This one takes a filename and size and uses the file specified
// to back the vector.
template<class EltT>
GrowableVector<EltT>::GrowableVector(
    AddPointOptions add_point_options,
    std::wstring filename,
    size_t initial_size)
    : GrowableVector{ add_point_options, filename }
{
    m_UsedSizeInElts = initial_size;
    m_CapacityInElts = initial_size;
}

template<class EltT>
GrowableVector<EltT>::~GrowableVector() {
    // File is also deleted via FILE_FLAG_DELETE_ON_CLOSE if needed
    CloseMapping(true);
}

template<class EltT>
EltT& GrowableVector<EltT>::operator[](size_t index) {
    return m_Data[index];
}

template<class EltT>
const EltT& GrowableVector<EltT>::operator[](size_t index) const {
    return m_Data[index];
}

template<class EltT>
void GrowableVector<EltT>::PushBack(const EltT& val) {
    const auto grow_by_elts = 128 * 1024;

    if (m_UsedSizeInElts == m_CapacityInElts) {
        MutableReserveKeepFileSize(m_CapacityInElts + grow_by_elts);
    }

    m_Data[m_UsedSizeInElts] = val;
    m_UsedSizeInElts++;
}

template<class EltT>
void GrowableVector<EltT>::PopBack() {
    if (m_UsedSizeInElts > 0) {
        m_UsedSizeInElts--;
    }
}

template<class EltT>
EltT& GrowableVector<EltT>::Back() {
    return m_Data[m_UsedSizeInElts - 1];
}

template<class EltT>
const EltT& GrowableVector<EltT>::Back() const {
    return m_Data[m_UsedSizeInElts - 1];
}

template<class EltT>
void GrowableVector<EltT>::Clear() {
    m_UsedSizeInElts = 0;
}

template<class EltT>
void GrowableVector<EltT>::CloseMapping(bool CloseFileToo) {
    if (!UsingAnonymous()) {
        if (m_Data != nullptr) {
            UnmapViewOfFile(m_Data);
            m_Data = nullptr;
        }
    }
    else {
        if (m_Data != nullptr) {
            VirtualFree(m_Data, 0, MEM_RELEASE);
            m_Data = nullptr;
        }
    }

    if (m_MappedFile != nullptr) {
        CloseHandle(m_MappedFile);
        m_MappedFile = nullptr;
    }

    if (CloseFileToo) {
        if (m_FileHandle != nullptr) {
            CloseHandle(m_FileHandle);
            m_FileHandle = nullptr;
        }
    }

    // m_UsedSizeInElts is unchanged.
    m_CapacityInElts = 0;
}

template<class EltT>
void GrowableVector<EltT>::Trim() {
    // TODO No-op for now.
}

template<class EltT>
AddPointOptions GrowableVector<EltT>::GetAddPointOptions() const {
    return m_AddPointOptions;
}

template<class EltT>
bool GrowableVector<EltT>::MutableReserveKeepFileSize(size_t capacity) {
    if (!UsingAnonymous()) {
        return MutableFileCommit(capacity);
    }
    else {
        return MutableAnonymousCommit(capacity);
    }
}

template<class EltT>
bool GrowableVector<EltT>::MutableResize(size_t capacity, size_t size) {
    bool ret = MutableReserveKeepFileSize(capacity);
    if (ret) {
        m_UsedSizeInElts = size;
    }

    return ret;
}


template<class EltT>
bool GrowableVector<EltT>::MutableResize(size_t size) {
    return MutableResize(size, size);
}

template<class EltT>
bool GrowableVector<EltT>::UsingAnonymous() const {
    return m_AddPointOptions == AddPointOptions::DontSave;
}

template<class EltT>
bool GrowableVector<EltT>::MutableFileCommit(size_t capacity) {
    if (capacity <= m_CapacityInElts) {
        return true;
    }

    CloseMapping(false);

    DWORD last_error = 0;

    if (m_FileHandle == nullptr) {
        auto attributes = FILE_ATTRIBUTE_NORMAL;
        if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
            attributes |= FILE_FLAG_DELETE_ON_CLOSE;
            attributes |= FILE_ATTRIBUTE_TEMPORARY;
        }

        DWORD desired_access = GENERIC_READ | GENERIC_WRITE;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            desired_access = GENERIC_READ;
        }

        DWORD open_mode = OPEN_ALWAYS;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            open_mode = OPEN_EXISTING;
        }

        m_FileHandle = CreateFile(m_Filename.c_str(),
            desired_access,
            FILE_SHARE_READ,
            nullptr,
            open_mode,
            attributes,
            nullptr);
        if (m_FileHandle == INVALID_HANDLE_VALUE) {
            m_FileHandle = nullptr;
            auto err = GetLastError();
            std::wstring err_str = L"Failed to open file for reading 2: ";
            err_str += std::to_wstring(err);
            ::MessageBox(nullptr, err_str.c_str(), L"", MB_OK | MB_APPLMODAL);
            return false;
        }

        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            last_error = ERROR_ALREADY_EXISTS;
        }
        else {
            // This should be either 0 or ERROR_ALREADY_EXISTS given that
            // the CreateFile call was evidently successful
            last_error = GetLastError();
        }
    }
    else {
        // The file must be there because it's open
        last_error = ERROR_ALREADY_EXISTS;
        assert(m_FileHandle != nullptr);
    }


    if (last_error == 0 || last_error == ERROR_ALREADY_EXISTS) {
        static_assert(sizeof(size_t) == sizeof(uint64_t), "!");
        static_assert(sizeof(DWORD) == sizeof(uint32_t), "!");
        uint64_t total_new_size = capacity * sizeof(EltT);
        DWORD high = total_new_size >> 32;
        DWORD low = total_new_size & 0xFFFFFFFF;

        LARGE_INTEGER existing_file_size{};
        if (last_error == ERROR_ALREADY_EXISTS) {
            BOOL ret = GetFileSizeEx(m_FileHandle, &existing_file_size);
            if (ret == FALSE) {
                ::MessageBox(nullptr, L"Failed to get file size", L"", MB_OK | MB_APPLMODAL);
                return false;
            }

            // Note: by using the file size to find the used size, the implication
            // is that resizing the vector only takes place once the vector is full.
            // This is not the case for anonymous memory.
            // The capacity/used size distinction is important as always.
            m_UsedSizeInElts = existing_file_size.QuadPart / sizeof(EltT);

            if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
                high = existing_file_size.HighPart;
                low = existing_file_size.LowPart;
                m_CapacityInElts = m_UsedSizeInElts;
            }
            else {
                m_CapacityInElts = capacity;
            }
        }
        else {
            m_UsedSizeInElts = 0;
            m_CapacityInElts = capacity;
        }

        DWORD protect = PAGE_READWRITE;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            protect = PAGE_READONLY;
        }

        m_MappedFile = CreateFileMapping(
            m_FileHandle,
            nullptr,
            protect,
            high,
            low,
            nullptr);
        if (m_MappedFile == nullptr) {
            auto err = GetLastError();
            std::wstring err_str = L"Failed to create file mapping: ";
            err_str += std::to_wstring(err);
            ::MessageBox(nullptr, err_str.c_str(), L"", MB_OK | MB_APPLMODAL);
            return false;
        }

        DWORD desired_access = FILE_MAP_READ | FILE_MAP_WRITE;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            desired_access = FILE_MAP_READ;
        }

        m_Data = static_cast<EltT*>(MapViewOfFile(m_MappedFile, desired_access, 0, 0, 0));
        if (m_Data == nullptr) {
            ::MessageBox(nullptr, L"Failed to map view of file", L"", MB_OK | MB_APPLMODAL);
            return false;
        }
    }
    else {
        auto ret = GetLastError();
        std::wstring err = L"Failed to open file: ";
        err += std::to_wstring(ret);
        ::MessageBox(nullptr, err.c_str(), L"", MB_OK | MB_APPLMODAL);
        return false;
    }

    return true;
}

template<class EltT>
bool GrowableVector<EltT>::MutableAnonymousCommit(size_t capacity) {
    // Returned value is in KB so convert to bytes.
    if (m_Data == nullptr) {
        if (UsingAnonymous()) {
            MutableReserve(m_PhysicalMemoryCapacityKB * 1024);
        }
    }

    if (capacity > m_CapacityInElts) {
        // If we're using anonymous memory, we need to commit the memory.
        // Use VirtualAlloc to allocate additional space for the vector.
        // The returned pointer should be the same as the original pointer.

        auto res = VirtualAlloc(
            m_Data,
            capacity * sizeof(EltT),
            MEM_COMMIT,
            PAGE_READWRITE
        );

        if (m_Data == nullptr) {
            std::wstring err = L"Failed to allocate memory: ";
            auto code = GetLastError();
            err += std::to_wstring(code);

            ::MessageBox(nullptr, err.c_str(), L"", MB_OK | MB_APPLMODAL);
            return false;
        }
        else if (m_Data != res) {
            std::wstring err = L"VirtualAlloc returned a different pointer :(";
            err += std::to_wstring(reinterpret_cast<uint64_t>(m_Data));
            err += L" vs ";
            err += std::to_wstring(reinterpret_cast<uint64_t>(res));
            err += L" :(.  Code: ";
            auto code = GetLastError();
            err += std::to_wstring(code);
            ::MessageBox(nullptr, err.c_str(), L"", MB_OK | MB_APPLMODAL);
            return false;
        }

        m_Data = static_cast<EltT*>(res);
        m_CapacityInElts = capacity;
    }

    return true;
}

template<class EltT>
void GrowableVector<EltT>::MutableReserve(size_t new_reserved_bytes)
{
    assert(UsingAnonymous());  // This function is only for anonymous memory.

    auto res = VirtualAlloc(
        m_Data,
        new_reserved_bytes,
        MEM_RESERVE,
        PAGE_READWRITE
    );

    if (res == nullptr) {
        std::wstring err = L"Failed to reserve memory: ";
        auto code = GetLastError();
        err += std::to_wstring(code);

        ::MessageBox(nullptr, err.c_str(), L"", MB_OK | MB_APPLMODAL);
        return;
    }

    m_Data = static_cast<EltT*>(res);
}


#define InstantiateLAInfoDeepGrowableVector(IterType, T, SubType, PExtras) \
    template class GrowableVector<LAInfoDeep<IterType, T, SubType, PExtras>>

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::Disable);

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::Bad);

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::Disable);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::Bad);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::EnableCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);


#define InstantiateGrowableVector(EltT) template class GrowableVector<EltT>

InstantiateGrowableVector(LAStageInfo<uint32_t>);
InstantiateGrowableVector(LAStageInfo<uint64_t>);

#define InstantiateGPUReferenceIterGrowableVector(T, PExtras) template class GrowableVector<GPUReferenceIter<T, PExtras>>

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::EnableCompression);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::EnableCompression);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::EnableCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::EnableCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
