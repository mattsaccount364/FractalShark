#include "stdafx.h"
#include "Vectors.h"

#include "Fractal.h"
#include "GPU_Render.h"
#include "HDRFloat.h"
#include "LAInfoDeep.h"

#include "Exceptions.h"
#include "Callstacks.h"

#include <assert.h>
#include <combaseapi.h>


#pragma comment(lib, "ntdll")

typedef uint32_t NTSTATUS;

typedef enum _SECTION_INHERIT {
    ViewShare = 1,
    ViewUnmap = 2
} SECTION_INHERIT;

typedef struct _LSA_UNICODE_STRING { USHORT Length;	USHORT MaximumLength; PWSTR  Buffer; } UNICODE_STRING, *PUNICODE_STRING;
typedef struct _OBJECT_ATTRIBUTES { ULONG Length; HANDLE RootDirectory; PUNICODE_STRING ObjectName; ULONG Attributes; PVOID SecurityDescriptor;	PVOID SecurityQualityOfService; } OBJECT_ATTRIBUTES, *POBJECT_ATTRIBUTES;
typedef struct _CLIENT_ID { PVOID UniqueProcess; PVOID UniqueThread; } CLIENT_ID, *PCLIENT_ID;
using myNtCreateSection = NTSTATUS(NTAPI *)(
    OUT PHANDLE SectionHandle,
    IN ULONG DesiredAccess,
    IN POBJECT_ATTRIBUTES ObjectAttributes OPTIONAL,
    IN PLARGE_INTEGER MaximumSize OPTIONAL,
    IN ULONG PageAttributess,
    IN ULONG SectionAttributes,
    IN HANDLE FileHandle OPTIONAL);
using myNtMapViewOfSection = NTSTATUS(NTAPI *)(
    HANDLE SectionHandle,
    HANDLE ProcessHandle,
    PVOID *BaseAddress,
    ULONG_PTR ZeroBits,
    SIZE_T CommitSize,
    PLARGE_INTEGER SectionOffset,
    PSIZE_T ViewSize,
    DWORD InheritDisposition,
    ULONG AllocationType,
    ULONG Win32Protect);
using myNtExtendSection = NTSTATUS(NTAPI *)(
    IN HANDLE               SectionHandle,
    IN PLARGE_INTEGER       NewSectionSize);

static myNtCreateSection fNtCreateSection;
static myNtMapViewOfSection fNtMapViewOfSection;
static myNtExtendSection fNtExtendSection;

void VectorStaticInit() {
    HMODULE hNtDll = GetModuleHandle(L"ntdll.dll");

    if (hNtDll == nullptr) {
        throw FractalSharkSeriousException("Failed to get handle to ntdll.dll");
    }

    fNtCreateSection = (myNtCreateSection)GetProcAddress(hNtDll, "NtCreateSection");
    fNtMapViewOfSection = (myNtMapViewOfSection)GetProcAddress(hNtDll, "NtMapViewOfSection");
    fNtExtendSection = (myNtExtendSection)GetProcAddress(hNtDll, "NtExtendSection");
}


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
    case GrowableVectorTypes::ImaginaFile:
        return L".im";
    case GrowableVectorTypes::DebugOutput:
        return L".debug.txt";
    case GrowableVectorTypes::ItersMemoryContainer:
        return L".iters";
    default:
        assert(false);
        return L".error";
    }
}

template<class EltT>
EltT *GrowableVector<EltT>::GetData() const {
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
GrowableVector<EltT>::GrowableVector(GrowableVector<EltT> &&other) noexcept :
    m_FileHandle{ other.m_FileHandle },
    m_MappedFile{ other.m_MappedFile },
    m_UsedSizeInElts{ other.m_UsedSizeInElts },
    m_CapacityInElts{ other.m_CapacityInElts },
    m_Data{ other.m_Data },
    m_AddPointOptions{ other.m_AddPointOptions },
    m_Filename{},
    m_PhysicalMemoryCapacityKB{ other.m_PhysicalMemoryCapacityKB } {

    wcscpy_s(m_Filename, other.m_Filename);

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_Data = nullptr;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    memset(other.m_Filename, 0, sizeof(other.m_Filename));
    other.m_PhysicalMemoryCapacityKB = 0;
}

template<class EltT>
GrowableVector<EltT> &GrowableVector<EltT>::operator=(GrowableVector<EltT> &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    CloseMapping();

    m_FileHandle = other.m_FileHandle;
    m_MappedFile = other.m_MappedFile;
    m_UsedSizeInElts = other.m_UsedSizeInElts;
    m_CapacityInElts = other.m_CapacityInElts;
    m_Data = other.m_Data;
    m_AddPointOptions = other.m_AddPointOptions;
    wcscpy_s(m_Filename, other.m_Filename);
    m_PhysicalMemoryCapacityKB = other.m_PhysicalMemoryCapacityKB;

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_Data = nullptr;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    memset(other.m_Filename, 0, sizeof(other.m_Filename));
    other.m_PhysicalMemoryCapacityKB = 0;

    return *this;
}

template<class EltT>
GrowableVector<EltT>::GrowableVector()
    : GrowableVector(AddPointOptions::DontSave, L"") {
    // Default to anonymous memory.
}

template<class EltT>
GrowableVector<EltT>::GrowableVector(
    AddPointOptions add_point_options,
    const wchar_t *filename)
    : m_FileHandle{},
    m_MappedFile{},
    m_UsedSizeInElts{},
    m_CapacityInElts{},
    m_Data{},
    m_AddPointOptions{ add_point_options },
    m_Filename{},
    m_PhysicalMemoryCapacityKB{} {

    wcscpy_s(m_Filename, filename);

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


// The constructor takes the file to open or create
// It maps enough memory to accomodate the provided orbit size.
template<class EltT>
GrowableVector<EltT>::GrowableVector(
    AddPointOptions add_point_options,
    std::wstring filename)
    : GrowableVector{ add_point_options, filename.c_str() } {
}

// This one takes a filename and size and uses the file specified
// to back the vector.
template<class EltT>
GrowableVector<EltT>::GrowableVector(
    AddPointOptions add_point_options,
    std::wstring filename,
    size_t initial_size)
    : GrowableVector{ add_point_options, filename } {
    m_UsedSizeInElts = initial_size;
    m_CapacityInElts = initial_size;
}

template<class EltT>
GrowableVector<EltT>::~GrowableVector() {
    // File is also deleted via FILE_FLAG_DELETE_ON_CLOSE if needed
    CloseMapping();
}

template<class EltT>
EltT &GrowableVector<EltT>::operator[](size_t index) {
    return m_Data[index];
}

template<class EltT>
const EltT &GrowableVector<EltT>::operator[](size_t index) const {
    return m_Data[index];
}

template<class EltT>
void GrowableVector<EltT>::GrowVectorIfNeeded() {
    if (m_UsedSizeInElts == m_CapacityInElts) {
        static constexpr size_t GrowByAmount = 256 * 1024 * 1024 / sizeof(EltT);
        MutableReserveKeepFileSize(m_CapacityInElts + GrowByAmount);
    }
}

template<class EltT>
void GrowableVector<EltT>::GrowVectorByAmount(size_t amount) {
    MutableResize(m_CapacityInElts + amount);
}

template<class EltT>
void GrowableVector<EltT>::PushBack(const EltT &val) {
    GrowVectorIfNeeded();
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
EltT &GrowableVector<EltT>::Back() {
    return m_Data[m_UsedSizeInElts - 1];
}

template<class EltT>
const EltT &GrowableVector<EltT>::Back() const {
    return m_Data[m_UsedSizeInElts - 1];
}

template<class EltT>
void GrowableVector<EltT>::Clear() {
    m_UsedSizeInElts = 0;
}

template<class EltT>
void GrowableVector<EltT>::CloseMapping() {
    if (UsingAnonymous()) {
        if (m_Data != nullptr) {
            GlobalCallstacks->LogDeallocCallstack(m_Data);
            auto res = VirtualFree(m_Data, 0, MEM_RELEASE);
            if (res == FALSE) {
                std::string err_str = "Failed to free memory: ";
                auto code = GetLastError();
                err_str += std::to_string(code);
                throw FractalSharkSeriousException(err_str);
            }

            m_Data = nullptr;
        }

        // m_UsedSizeInElts is unchanged.
        m_CapacityInElts = 0;
    } else {
        if (m_Data != nullptr) {
            UnmapViewOfFile(m_Data);
            m_Data = nullptr;
        }
    }

    if (m_MappedFile != nullptr) {
        CloseHandle(m_MappedFile);
        m_MappedFile = nullptr;
    }

    Trim();

    if (m_FileHandle != nullptr) {
        CloseHandle(m_FileHandle);
        m_FileHandle = nullptr;
    }

    // m_UsedSizeInElts is unchanged.
    m_CapacityInElts = 0;
}

template<class EltT>
void GrowableVector<EltT>::TrimEnableWithoutSave() {
    if (m_AddPointOptions != AddPointOptions::EnableWithoutSave) {
        return;
    }
    if (m_MappedFile == nullptr) {
        return;
    }

    const void *originalData = m_Data;

    // First unmap the existing view:
    UnmapViewOfFile(m_Data);

    // Don't set m_Data to null so we reuse the same address
    SIZE_T viewSize = m_UsedSizeInElts * sizeof(EltT);
    auto status2 = fNtMapViewOfSection(
        m_MappedFile,
        GetCurrentProcess(),
        (PVOID *)&m_Data,
        0,
        0,
        0,
        &viewSize,
        ViewUnmap,
        MEM_RESERVE,
        PAGE_READWRITE);
    if (status2 > 0) {
        std::string err_str = "Failed to map view of section: " + std::to_string(status2);
        throw FractalSharkSeriousException(err_str);
    }

    if (originalData != m_Data) {
        std::string err = "NtMapViewOfSection returned a different pointer :(";
        err += std::to_string(reinterpret_cast<uint64_t>(m_Data));
        err += " vs ";
        err += std::to_string(reinterpret_cast<uint64_t>(originalData));
        err += " :(";
        throw FractalSharkSeriousException(err);
    }
}

template<class EltT>
void GrowableVector<EltT>::TrimEnableWithSave() {
    if (m_AddPointOptions != AddPointOptions::EnableWithSave) {
        return;
    }

    // Set the location in the file to match the used size
    // Use 64-bit file size functions because the file could be large.
    // Only do it when we're actually keeping the result.

    LARGE_INTEGER distanceToMove{};
    distanceToMove.QuadPart = m_UsedSizeInElts * sizeof(EltT);
    auto result = SetFilePointerEx(m_FileHandle, distanceToMove, nullptr, FILE_BEGIN);
    if (result) {
        SetEndOfFile(m_FileHandle);
    }
}

template<class EltT>
void GrowableVector<EltT>::Trim() {
    TrimEnableWithoutSave();
    TrimEnableWithSave();
}

template<class EltT>
AddPointOptions GrowableVector<EltT>::GetAddPointOptions() const {
    return m_AddPointOptions;
}

template<class EltT>
void GrowableVector<EltT>::MutableReserveKeepFileSize(size_t capacity) {
    if (!UsingAnonymous()) {
        MutableFileCommit(capacity);
    } else {
        MutableAnonymousCommit(capacity);
    }
}

template<class EltT>
void GrowableVector<EltT>::MutableResize(size_t capacity, size_t size) {
    MutableReserveKeepFileSize(capacity);
    m_UsedSizeInElts = size;
}


template<class EltT>
void GrowableVector<EltT>::MutableResize(size_t size) {
    MutableResize(size, size);
}

template<class EltT>
bool GrowableVector<EltT>::UsingAnonymous() const {
    return m_AddPointOptions == AddPointOptions::DontSave;
}

template<class EltT>
uint32_t GrowableVector<EltT>::InternalOpenFile() {
    DWORD lastError = 0;

    if (m_FileHandle == nullptr) {
        auto attributes = FILE_ATTRIBUTE_NORMAL;
        if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
            attributes |= FILE_FLAG_DELETE_ON_CLOSE;
            attributes |= FILE_ATTRIBUTE_TEMPORARY;
        }

        DWORD desired_access = GENERIC_READ | GENERIC_WRITE;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            //desired_access = GENERIC_READ; // TODO
        }

        DWORD open_mode = OPEN_ALWAYS;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            open_mode = OPEN_EXISTING;
        }

        if (wcscmp(m_Filename, L"") == 0) {
            // Fill a wide string with a random guid
            GUID guid;
            auto res = CoCreateGuid(&guid);
            if (res != S_OK) {
                std::string err_str = "Failed to create GUID: ";
                err_str += std::to_string(res);
                throw FractalSharkSeriousException(err_str);
            }

            wchar_t guid_str[40];
            auto res2 = StringFromGUID2(guid, guid_str, 40);
            if (res2 == 0) {
                std::string err_str = "Failed to convert GUID to string: ";
                err_str += std::to_string(res2);
                throw FractalSharkSeriousException(err_str);
            }

            // Generate a temporary filename to a non-existent file
            // in the current directory using GetTempFileName
            wcscpy_s(m_Filename, guid_str);
        }

        m_FileHandle = CreateFile(m_Filename,
            desired_access,
            FILE_SHARE_READ,
            nullptr,
            open_mode,
            attributes,
            nullptr);
        if (m_FileHandle == INVALID_HANDLE_VALUE) {
            m_FileHandle = nullptr;
            auto err = GetLastError();
            std::string err_str = "Failed to open file for reading 2: ";
            err_str += std::to_string(err);
            throw FractalSharkSeriousException(err_str);
        }

        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            lastError = ERROR_ALREADY_EXISTS;
        } else {
            // This should be either 0 or ERROR_ALREADY_EXISTS given that
            // the CreateFile call was evidently successful
            lastError = GetLastError();
        }
    } else {
        // The file must be there because it's open
        lastError = ERROR_ALREADY_EXISTS;
        assert(m_FileHandle != nullptr);
    }

    return lastError;
}

template<class EltT>
void GrowableVector<EltT>::MutableFileResizeOpen(size_t capacity) {
    m_CapacityInElts = capacity;

    // Convert m_CapacityInElts to LARGE_INTEGER NewSize:
    LARGE_INTEGER NewSize;
    NewSize.QuadPart = m_CapacityInElts * sizeof(EltT);

    // this call extend file, section and view size
    auto status = fNtExtendSection(m_MappedFile, &NewSize);
    if (status > 0) {
        std::string err_str = "Failed to extend section: " + std::to_string(status);
        throw FractalSharkSeriousException(err_str);
    }
}

template<class EltT>
void GrowableVector<EltT>::MutableFileCommit(size_t capacity) {
    if (capacity <= m_CapacityInElts) {
        return;
    }

    if (m_FileHandle != nullptr && m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        MutableFileResizeOpen(capacity);
        return;
    }

    DWORD lastError = InternalOpenFile();

    // Check all the things that could have gone wrong
    if (m_FileHandle == nullptr ||
        m_FileHandle == INVALID_HANDLE_VALUE ||
        (lastError != 0 && lastError != ERROR_ALREADY_EXISTS)) {

        auto ret = GetLastError();
        std::string err_str = "Failed to open file: " + std::to_string(ret);
        throw FractalSharkSeriousException(err_str);
    }

    static_assert(sizeof(size_t) == sizeof(uint64_t), "!");
    static_assert(sizeof(DWORD) == sizeof(uint32_t), "!");
    uint64_t total_new_size = capacity * sizeof(EltT);
    DWORD high = total_new_size >> 32;
    DWORD low = total_new_size & 0xFFFFFFFF;

    LARGE_INTEGER existing_file_size{};
    if (lastError == ERROR_ALREADY_EXISTS) {
        BOOL ret = GetFileSizeEx(m_FileHandle, &existing_file_size);
        if (ret == FALSE) {
            std::string err_str = "Failed to get file size";
            throw FractalSharkSeriousException(err_str);
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
        } else {
            m_CapacityInElts = capacity;
        }
    } else {
        m_UsedSizeInElts = 0;
        m_CapacityInElts = capacity;
    }

    DWORD sectionPageProtection = PAGE_READWRITE;
    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        //sectionPageProtection = PAGE_READONLY; // TODO
    }

    /*
    m_MappedFile = CreateFileMapping(
        m_FileHandle,
        nullptr,
        sectionPageProtection,
        high,
        low,
        nullptr);
    if (m_MappedFile == nullptr) {
        auto err = GetLastError();
        std::string err_str = "Failed to create file mapping: ";
        err_str += std::to_string(err);
        throw FractalSharkSeriousException(err_str);
    }

    DWORD desired_access = FILE_MAP_READ | FILE_MAP_WRITE;
    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        desired_access = FILE_MAP_READ;
    }

    m_Data = static_cast<EltT*>(MapViewOfFile(m_MappedFile, desired_access, 0, 0, 0));
    if (m_Data == nullptr) {
        auto err_str = "Failed to map view of file";
        throw FractalSharkSeriousException(err_str);
    }
    */

    DWORD desired_access = SECTION_ALL_ACCESS;
    //  DWORD desired_access = SECTION_MAP_READ | SECTION_MAP_WRITE | SECTION_EXTEND_SIZE;
    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        //desired_access = SECTION_MAP_READ | SECTION_EXTEND_SIZE; // TODO
    }

    // set for demo intial size of section to 2 byte
    // NtCreateSection rounds this value up to the nearest multiple of PAGE_SIZE. 
    // however this will be have effect for file size exactly, if current file size less than this value
    LARGE_INTEGER initialSize;
    ULONG allocationAttributes;
    ULONG allocationType;
    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        initialSize.QuadPart =
            m_CapacityInElts * sizeof(EltT);
        allocationAttributes = SEC_RESERVE;
        allocationType = MEM_RESERVE;
    } else {
        initialSize.QuadPart = 0;
        allocationAttributes = SEC_RESERVE;
        allocationType = MEM_RESERVE;
    }

    DWORD status = fNtCreateSection(
        &m_MappedFile,
        desired_access,
        0,
        &initialSize,
        sectionPageProtection,
        allocationAttributes,
        m_FileHandle);

    //we can close file handle now
    //CloseHandle(hFile);

    if (status > 0) {
        std::string err_str = "Failed to create section: " + std::to_string(status);
        throw FractalSharkSeriousException(err_str);
    }

    SIZE_T viewSize;
    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        viewSize = m_PhysicalMemoryCapacityKB * 1024;
    } else {
        viewSize = existing_file_size.QuadPart;
    }

    // m_Data = MapViewOfFile(m_MappedFile, FILE_MAP_RESERVE|FILE_MAP_READ|FILE_MAP_WRITE, 0, 0, ViewSize);
    // note MEM_RESERVE
    const void *OriginalData = m_Data;
    status = fNtMapViewOfSection(
        m_MappedFile,
        GetCurrentProcess(),
        (PVOID *)&m_Data,
        0,
        0,
        0,
        &viewSize,
        ViewUnmap,
        allocationType,
        sectionPageProtection);
    if (status > 0) {
        std::string err_str = "Failed to map view of section: " + std::to_string(status);
        throw FractalSharkSeriousException(err_str);
    }

    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {

        // Debug check:
        if (OriginalData != nullptr &&
            OriginalData != m_Data) {

            std::string err = "NtMapViewOfSection returned a different pointer :(";
            err += std::to_string(reinterpret_cast<uint64_t>(m_Data));
            err += " vs ";
            err += std::to_string(reinterpret_cast<uint64_t>(OriginalData));
            err += " :(";
            throw FractalSharkSeriousException(err);
        }

        // Convert m_CapacityInElts to LARGE_INTEGER NewSize:
        LARGE_INTEGER NewSize;
        NewSize.QuadPart = m_CapacityInElts * sizeof(EltT);

        // this call extend file, section and view size
        status = fNtExtendSection(m_MappedFile, &NewSize);
        if (status > 0) {
            std::string err_str = "Failed to extend section: " + std::to_string(status);
            throw FractalSharkSeriousException(err_str);
        }
    }
}

template<class EltT>
void GrowableVector<EltT>::MutableAnonymousCommit(size_t capacity) {
    // Returned value is in KB so convert to bytes.
    if (m_Data == nullptr) {
        assert(UsingAnonymous());
        MutableReserve(m_PhysicalMemoryCapacityKB * 1024);
    }

    if (capacity > m_CapacityInElts) {
        // If we're using anonymous memory, we need to commit the memory.
        // Use VirtualAlloc to allocate additional space for the vector.
        // The returned pointer should be the same as the original pointer.

        const auto bytesCount = capacity * sizeof(EltT);
        auto res = VirtualAlloc(
            m_Data,
            bytesCount,
            MEM_COMMIT,
            PAGE_READWRITE
        );

        GlobalCallstacks->LogAllocCallstack(bytesCount, res);

        if (m_Data == nullptr) {
            std::string err_str = "Failed to allocate memory: ";
            auto code = GetLastError();
            err_str += std::to_string(code);
            throw FractalSharkSeriousException(err_str);
        } else if (m_Data != res) {
            std::string err = "VirtualAlloc returned a different pointer :(";
            err += std::to_string(reinterpret_cast<uint64_t>(m_Data));
            err += " vs ";
            err += std::to_string(reinterpret_cast<uint64_t>(res));
            err += " :(.  Code: ";
            auto code = GetLastError();
            err += std::to_string(code);
            throw FractalSharkSeriousException(err);
        }

        m_Data = static_cast<EltT *>(res);
        m_CapacityInElts = capacity;
    }
}

template<class EltT>
void GrowableVector<EltT>::MutableReserve(size_t new_reserved_bytes) {
    assert(UsingAnonymous());  // This function is only for anonymous memory.

    auto res = VirtualAlloc(
        m_Data,
        new_reserved_bytes,
        MEM_RESERVE,
        PAGE_READWRITE
    );

    GlobalCallstacks->LogReserveCallstack(new_reserved_bytes, res);

    if (res == nullptr) {
        std::wstring err = L"Failed to reserve memory: ";
        auto code = GetLastError();
        err += std::to_wstring(code);

        ::MessageBox(nullptr, err.c_str(), L"", MB_OK | MB_APPLMODAL);
        return;
    }

    m_Data = static_cast<EltT *>(res);
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

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

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

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);


#define InstantiateGrowableVector(EltT) template class GrowableVector<EltT>

InstantiateGrowableVector(LAStageInfo<uint32_t>);
InstantiateGrowableVector(LAStageInfo<uint64_t>);
InstantiateGrowableVector(uint8_t);
InstantiateGrowableVector(uint16_t);
InstantiateGrowableVector(uint32_t);
InstantiateGrowableVector(uint64_t);

#define InstantiateGPUReferenceIterGrowableVector(T, PExtras) template class GrowableVector<GPUReferenceIter<T, PExtras>>

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::Disable);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::SimpleCompression);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::SimpleCompression);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::SimpleCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::SimpleCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::MaxCompression);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
