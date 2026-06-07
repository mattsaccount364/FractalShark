#ifndef _WIN32

#include "Callstacks.h"
#include "DbgHeap.h"
#include "EarlyCommandLine.h"
#include "Exceptions.h"
#include "GPU_ReferenceIter.h"
#include "HDRFloat.h"
#include "LAInfoDeep.h"
#include "Vectors.h"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// -------------------------------------------------------------------------
// Helper: convert wchar_t (UTF-32 on Linux) to UTF-8.
// -------------------------------------------------------------------------
namespace {

bool
AppendUtf8(char *dst, size_t dstSize, size_t &offset, uint32_t c)
{
    auto appendByte = [&](char b) {
        if (offset + 1 >= dstSize) {
            return false;
        }
        dst[offset++] = b;
        return true;
    };

    if (c < 0x80) {
        return appendByte(static_cast<char>(c));
    }
    if (c < 0x800) {
        return appendByte(static_cast<char>(0xC0 | (c >> 6))) &&
               appendByte(static_cast<char>(0x80 | (c & 0x3F)));
    }
    if (c < 0x10000) {
        return appendByte(static_cast<char>(0xE0 | (c >> 12))) &&
               appendByte(static_cast<char>(0x80 | ((c >> 6) & 0x3F))) &&
               appendByte(static_cast<char>(0x80 | (c & 0x3F)));
    }

    return appendByte(static_cast<char>(0xF0 | (c >> 18))) &&
           appendByte(static_cast<char>(0x80 | ((c >> 12) & 0x3F))) &&
           appendByte(static_cast<char>(0x80 | ((c >> 6) & 0x3F))) &&
           appendByte(static_cast<char>(0x80 | (c & 0x3F)));
}

bool
WideToUtf8Buffer(const wchar_t *wide, char *dst, size_t dstSize)
{
    if (dstSize == 0) {
        return false;
    }

    size_t offset = 0;
    if (wide == nullptr) {
        dst[0] = '\0';
        return true;
    }

    for (const wchar_t *p = wide; *p != L'\0'; ++p) {
        if (!AppendUtf8(dst, dstSize, offset, static_cast<uint32_t>(*p))) {
            dst[0] = '\0';
            return false;
        }
    }

    dst[offset] = '\0';
    return true;
}

size_t
MappingViewSize(void *mappedFile)
{
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(mappedFile));
}

void *
MappingViewState(size_t viewSize)
{
    return reinterpret_cast<void *>(static_cast<uintptr_t>(viewSize));
}

} // anonymous namespace

// Portable replacement for Microsoft's wcscpy_s. Copies up to N-1 wide
// characters and always null-terminates the destination.
template <size_t N>
static inline void
wcscpy_s(wchar_t (&dst)[N], const wchar_t *src)
{
    if (src == nullptr) {
        dst[0] = L'\0';
        return;
    }
    size_t i = 0;
    for (; i + 1 < N && src[i] != L'\0'; ++i) {
        dst[i] = src[i];
    }
    dst[i] = L'\0';
}

// -------------------------------------------------------------------------
// VectorStaticInit — no-op on Linux (no ntdll to load).
// -------------------------------------------------------------------------
void
VectorStaticInit()
{
}

// -------------------------------------------------------------------------
// GetFileExtension
// -------------------------------------------------------------------------
std::wstring
GetFileExtension(GrowableVectorTypes Type)
{
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

// =========================================================================
// Simple accessors (identical to Win32)
// =========================================================================

template <class EltT>
EltT *
GrowableVector<EltT>::GetData() const
{
    return m_Data;
}

template <class EltT>
std::wstring
GrowableVector<EltT>::GetFilename() const
{
    return m_Filename;
}

template <class EltT>
size_t
GrowableVector<EltT>::GetCapacity() const
{
    return m_CapacityInElts;
}

template <class EltT>
size_t
GrowableVector<EltT>::GetSize() const
{
    return m_UsedSizeInElts;
}

template <class EltT>
bool
GrowableVector<EltT>::ValidFile() const
{
    return m_Data != nullptr;
}

// =========================================================================
// Move constructor / assignment (identical to Win32)
// =========================================================================

template <class EltT>
GrowableVector<EltT>::GrowableVector(GrowableVector<EltT> &&other) noexcept
    : m_FileHandle{other.m_FileHandle}, m_MappedFile{other.m_MappedFile},
      m_UsedSizeInElts{other.m_UsedSizeInElts}, m_CapacityInElts{other.m_CapacityInElts},
      m_Data{other.m_Data}, m_AddPointOptions{other.m_AddPointOptions}, m_Filename{},
      m_PhysicalMemoryCapacityKB{other.m_PhysicalMemoryCapacityKB},
      m_OverrideViewSizeBytes{other.m_OverrideViewSizeBytes}
{

    wcscpy_s(m_Filename, other.m_Filename);

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_Data = nullptr;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    memset(other.m_Filename, 0, sizeof(other.m_Filename));
    other.m_PhysicalMemoryCapacityKB = 0;
    other.m_OverrideViewSizeBytes = 0;
}

template <class EltT>
GrowableVector<EltT> &
GrowableVector<EltT>::operator=(GrowableVector<EltT> &&other) noexcept
{
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
    m_OverrideViewSizeBytes = other.m_OverrideViewSizeBytes;

    other.m_FileHandle = nullptr;
    other.m_MappedFile = nullptr;
    other.m_UsedSizeInElts = 0;
    other.m_CapacityInElts = 0;
    other.m_Data = nullptr;
    other.m_AddPointOptions = AddPointOptions::DontSave;
    memset(other.m_Filename, 0, sizeof(other.m_Filename));
    other.m_PhysicalMemoryCapacityKB = 0;
    other.m_OverrideViewSizeBytes = 0;

    return *this;
}

// =========================================================================
// Constructors / destructor
// =========================================================================

template <class EltT>
GrowableVector<EltT>::GrowableVector() : GrowableVector(AddPointOptions::DontSave, L"")
{
    // Default to anonymous memory.
}

template <class EltT>
GrowableVector<EltT>::GrowableVector(AddPointOptions addPointOptions,
                                     const wchar_t *filename,
                                     size_t overrideViewSize)
    : m_FileHandle{}, m_MappedFile{}, m_UsedSizeInElts{}, m_CapacityInElts{}, m_Data{},
      m_AddPointOptions{addPointOptions}, m_Filename{}, m_PhysicalMemoryCapacityKB{},
      m_OverrideViewSizeBytes{overrideViewSize}
{

    wcscpy_s(m_Filename, filename);

    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || pageSize <= 0) {
        throw FractalSharkSeriousException("Failed to get system memory", false);
    }
    m_PhysicalMemoryCapacityKB = static_cast<size_t>(pages) * static_cast<size_t>(pageSize) / 1024;

    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        // 32 MB.  We're not allocating some function of RAM on disk.
        // It should be doing a sparse allocation but who knows.
        MutableFileCommit(1024);
    }
}

// This one takes a filename and size and uses the file specified
// to back the vector.
// The constructor takes the file to open or create
// It maps enough memory to accomodate the provided orbit size.
template <class EltT>
GrowableVector<EltT>::GrowableVector(AddPointOptions addPointOptions, const wchar_t *filename)
    : GrowableVector{addPointOptions, filename, 0}
{
}

template <class EltT> GrowableVector<EltT>::~GrowableVector()
{
    // File is also deleted via unlink-on-open if needed
    CloseMapping();
}

// =========================================================================
// Element access (identical to Win32)
// =========================================================================

template <class EltT>
EltT &
GrowableVector<EltT>::operator[](size_t index)
{
    return m_Data[index];
}

template <class EltT>
const EltT &
GrowableVector<EltT>::operator[](size_t index) const
{
    return m_Data[index];
}

template <class EltT>
void
GrowableVector<EltT>::GrowVectorIfNeeded()
{
    if (m_UsedSizeInElts == m_CapacityInElts) {
        static constexpr size_t GrowByAmount = 256 * 1024 * 1024 / sizeof(EltT);
        MutableReserveKeepFileSize(m_CapacityInElts + GrowByAmount);
    }
}

template <class EltT>
void
GrowableVector<EltT>::GrowVectorByAmount(size_t amount)
{
    MutableResize(m_CapacityInElts + amount);
}

template <class EltT>
void
GrowableVector<EltT>::PushBack(const EltT &val)
{
    GrowVectorIfNeeded();
    m_Data[m_UsedSizeInElts] = val;
    m_UsedSizeInElts++;
}

template <class EltT>
void
GrowableVector<EltT>::PopBack()
{
    if (m_UsedSizeInElts > 0) {
        m_UsedSizeInElts--;
    }
}

template <class EltT>
EltT &
GrowableVector<EltT>::Back()
{
    return m_Data[m_UsedSizeInElts - 1];
}

template <class EltT>
const EltT &
GrowableVector<EltT>::Back() const
{
    return m_Data[m_UsedSizeInElts - 1];
}

template <class EltT>
void
GrowableVector<EltT>::Clear()
{
    m_UsedSizeInElts = 0;
}

// =========================================================================
// AddPointOptions accessor
// =========================================================================

template <class EltT>
AddPointOptions
GrowableVector<EltT>::GetAddPointOptions() const
{
    return m_AddPointOptions;
}

// =========================================================================
// CloseMapping — tears down mmap and closes the file descriptor.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::CloseMapping()
{
    if (UsingAnonymous()) {
        if (m_Data != nullptr) {
            GlobalCallstacks->LogDeallocCallstack(m_Data);
            size_t sz = MappingViewSize(m_MappedFile);
            if (sz > 0) {
                munmap(m_Data, sz);
            }

            m_Data = nullptr;
        }

        // m_UsedSizeInElts is unchanged.
        m_CapacityInElts = 0;
    } else {
        if (m_Data != nullptr) {
            size_t sz = MappingViewSize(m_MappedFile);
            if (sz > 0) {
                munmap(m_Data, sz);
            }
            m_Data = nullptr;
        }
    }

    if (m_MappedFile != nullptr) {
        m_MappedFile = nullptr;
    }

    Trim();

    if (m_FileHandle != nullptr) {
        int fd = static_cast<int>(reinterpret_cast<intptr_t>(m_FileHandle));
        close(fd);
        m_FileHandle = nullptr;
    }

    // m_UsedSizeInElts is unchanged.
    m_CapacityInElts = 0;
}

// =========================================================================
// TrimEnableWithoutSave — shrink the mmap'd view via mremap.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::TrimEnableWithoutSave()
{
    if (m_AddPointOptions != AddPointOptions::EnableWithoutSave) {
        return;
    }
    if (m_MappedFile == nullptr) {
        return;
    }

    size_t newSize = m_UsedSizeInElts * sizeof(EltT);
    if (newSize == 0) {
        newSize = 1; // mremap doesn't accept 0
    }

    size_t currentSize = MappingViewSize(m_MappedFile);
    if (currentSize == 0 || newSize >= currentSize) {
        return;
    }

    const void *originalData = m_Data;

    // Shrink in-place (flags=0 means don't move)
    void *result = mremap(m_Data, currentSize, newSize, 0);
    if (result == MAP_FAILED) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to mremap view: %s", strerror(errno));
        throw FractalSharkSeriousException(buf, false);
    }

    m_Data = static_cast<EltT *>(result);
    m_MappedFile = MappingViewState(newSize);

    if (originalData != m_Data) {
        // mremap with flags=0 should not move the mapping
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "mremap returned a different pointer: %llu vs %llu",
                 (unsigned long long)reinterpret_cast<uint64_t>(m_Data),
                 (unsigned long long)reinterpret_cast<uint64_t>(originalData));
        throw FractalSharkSeriousException(buf, false);
    }
}

// =========================================================================
// TrimEnableWithSave — truncate the backing file to used size.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::TrimEnableWithSave()
{
    if (m_AddPointOptions != AddPointOptions::EnableWithSave) {
        return;
    }

    int fd = static_cast<int>(reinterpret_cast<intptr_t>(m_FileHandle));
    off_t usedBytes = static_cast<off_t>(m_UsedSizeInElts * sizeof(EltT));
    ftruncate(fd, usedBytes);
}

// =========================================================================
// Trim
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::Trim()
{
    TrimEnableWithoutSave();
    TrimEnableWithSave();
}

// =========================================================================
// Resize / reserve helpers (identical logic to Win32)
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::MutableReserveKeepFileSize(size_t capacity)
{
    if (!UsingAnonymous()) {
        MutableFileCommit(capacity);
    } else {
        MutableAnonymousCommit(capacity);
    }
}

template <class EltT>
void
GrowableVector<EltT>::MutableResize(size_t capacity, size_t size)
{
    MutableReserveKeepFileSize(capacity);
    m_UsedSizeInElts = size;
}

template <class EltT>
void
GrowableVector<EltT>::MutableResize(size_t size)
{
    MutableResize(size, size);
}

template <class EltT>
bool
GrowableVector<EltT>::UsingAnonymous() const
{
    return m_AddPointOptions == AddPointOptions::DontSave;
}

// =========================================================================
// InternalOpenFile — open (or create) the backing file via POSIX open().
// Returns 0 for a newly-created file or 183 (ERROR_ALREADY_EXISTS
// equivalent) when the file already existed.
// =========================================================================

template <class EltT>
uint32_t
GrowableVector<EltT>::InternalOpenFile()
{
    uint32_t lastError = 0;

    if (m_FileHandle == nullptr) {
        int oflags = O_RDWR | O_CREAT;
        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            oflags = O_RDWR; // no O_CREAT — file must exist
        } else if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
            oflags |= O_TRUNC;
        }

        if (wcscmp(m_Filename, L"") == 0) {
            // Generate a random filename from /dev/urandom.
            uint8_t bytes[16];
            int randomFd = open("/dev/urandom", O_RDONLY);
            if (randomFd < 0 ||
                read(randomFd, bytes, sizeof(bytes)) != static_cast<ssize_t>(sizeof(bytes))) {
                if (randomFd >= 0) {
                    close(randomFd);
                }
                throw FractalSharkSeriousException("Failed to generate random filename", false);
            }
            close(randomFd);

            wchar_t guid[40];
            swprintf(guid,
                     40,
                     L"{%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x}",
                     bytes[0],
                     bytes[1],
                     bytes[2],
                     bytes[3],
                     bytes[4],
                     bytes[5],
                     bytes[6],
                     bytes[7],
                     bytes[8],
                     bytes[9],
                     bytes[10],
                     bytes[11],
                     bytes[12],
                     bytes[13],
                     bytes[14],
                     bytes[15]);
            wcscpy_s(m_Filename, guid);
        }

        // Convert wide filename to UTF-8 for POSIX APIs.
        char u8path[4096];
        if (!WideToUtf8Buffer(m_Filename, u8path, sizeof(u8path))) {
            throw FractalSharkSeriousException("Failed to convert filename to UTF-8", false);
        }

        int fd = open(u8path, oflags, 0666);
        if (fd < 0) {
            auto err = errno;
            char buf[256];
            snprintf(buf, sizeof(buf), "Failed to open file: %d", err);
            throw FractalSharkSeriousException(buf, false);
        }

        m_FileHandle = reinterpret_cast<void *>(static_cast<intptr_t>(fd));

        // Delete-on-close: unlink now, data persists until fd is closed.
        if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
            unlink(u8path);
        }

        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            lastError = 183; // ERROR_ALREADY_EXISTS equivalent
        } else {
            // Check if file existed before by seeing if it had content.
            struct stat st;
            if (fstat(fd, &st) == 0 && st.st_size > 0) {
                lastError = 183;
            } else {
                lastError = 0;
            }
        }
    } else {
        // The file must be there because it's open.
        lastError = 183;
        assert(m_FileHandle != nullptr);
    }

    return lastError;
}

// =========================================================================
// MutableFileResizeOpen — extend backing file via ftruncate.
// The existing MAP_SHARED mapping sees new data for in-range pages.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::MutableFileResizeOpen(size_t capacity)
{
    const size_t capacityBytes = capacity * sizeof(EltT);
    if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
        const size_t viewSize = MappingViewSize(m_MappedFile);
        if (capacityBytes > viewSize) {
            throw FractalSharkSeriousException("GrowableVector sparse backing view exhausted", false);
        }
        m_CapacityInElts = capacity;
        return;
    }

    m_CapacityInElts = capacity;

    int fd = static_cast<int>(reinterpret_cast<intptr_t>(m_FileHandle));
    off_t newSize = static_cast<off_t>(capacityBytes);
    if (ftruncate(fd, newSize) != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to extend file: %s", strerror(errno));
        throw FractalSharkSeriousException(buf, false);
    }
}

// =========================================================================
// MutableFileCommit — create/extend the file-backed mapping.
//
// Flow mirrors the Win32 NtCreateSection/NtMapViewOfSection/NtExtendSection
// sequence, using ftruncate + mmap + ftruncate instead.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::MutableFileCommit(size_t capacity)
{
    if (capacity <= m_CapacityInElts) {
        return;
    }

    if (m_FileHandle != nullptr && m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        MutableFileResizeOpen(capacity);
        return;
    }

    uint32_t lastError = InternalOpenFile();

    // Check all the things that could have gone wrong.
    if (m_FileHandle == nullptr || (lastError != 0 && lastError != 183)) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to open file: %u", lastError);
        throw FractalSharkSeriousException(buf, false);
    }

    static_assert(sizeof(size_t) == sizeof(uint64_t), "!");
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(m_FileHandle));

    off_t existingFileSize = 0;
    if (lastError == 183) {
        struct stat st;
        if (fstat(fd, &st) != 0) {
            throw FractalSharkSeriousException("Failed to get file size", false);
        }
        existingFileSize = st.st_size;

        // Note: by using the file size to find the used size, the implication
        // is that resizing the vector only takes place once the vector is full.
        // This is not the case for anonymous memory.
        // The capacity/used size distinction is important as always.
        m_UsedSizeInElts = static_cast<size_t>(existingFileSize) / sizeof(EltT);

        if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
            m_CapacityInElts = m_UsedSizeInElts;
        } else {
            m_CapacityInElts = capacity;
        }
    } else {
        m_UsedSizeInElts = 0;
        m_CapacityInElts = capacity;
    }

    // Determine the view (mmap) size.
    size_t viewSize;
    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        if (m_OverrideViewSizeBytes > 0) {
            viewSize = m_OverrideViewSizeBytes;
        } else {
            viewSize = m_PhysicalMemoryCapacityKB * 1024;
        }
    } else {
        viewSize = static_cast<size_t>(existingFileSize);
    }

    const size_t capacityBytes = m_CapacityInElts * sizeof(EltT);
    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave && capacityBytes > viewSize) {
        throw FractalSharkSeriousException("GrowableVector capacity exceeds sparse backing view", false);
    }

    // Set the initial file size via ftruncate (analogous to NtCreateSection
    // with initialSize). Temporary file-backed vectors use the full sparse
    // backing file immediately; disk blocks are still allocated only as pages
    // are written.
    off_t initialSize;
    if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
        initialSize = static_cast<off_t>(viewSize);
    } else if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        initialSize = static_cast<off_t>(capacityBytes);
    } else {
        initialSize = existingFileSize;
    }
    if (ftruncate(fd, initialSize) != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to set initial file size: %s", strerror(errno));
        throw FractalSharkSeriousException(buf, false);
    }

    // mmap the file.
    const void *originalData = m_Data;
    void *mapped = mmap(nullptr, viewSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to mmap file: %s", strerror(errno));
        throw FractalSharkSeriousException(buf, false);
    }
    m_Data = static_cast<EltT *>(mapped);

    // Store the mapping size without allocating; this path is used while
    // bootstrapping the custom heap.
    m_MappedFile = MappingViewState(viewSize);

    if (m_AddPointOptions != AddPointOptions::OpenExistingWithSave) {
        // Debug check:
        if (originalData != nullptr && originalData != m_Data) {
            char buf[256];
            snprintf(buf,
                     sizeof(buf),
                     "mmap returned a different pointer: %llu vs %llu",
                     (unsigned long long)reinterpret_cast<uint64_t>(m_Data),
                     (unsigned long long)reinterpret_cast<uint64_t>(originalData));
            throw FractalSharkSeriousException(buf, false);
        }

        if (m_AddPointOptions == AddPointOptions::EnableWithoutSave) {
            return;
        }

        // Extend the file to its current logical capacity (analogous to NtExtendSection).
        if (ftruncate(fd, static_cast<off_t>(capacityBytes)) != 0) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Failed to extend file: %s", strerror(errno));
            throw FractalSharkSeriousException(buf, false);
        }
    }
}

// =========================================================================
// MutableAnonymousCommit — commit pages within the reserved region
// by changing protection from PROT_NONE to PROT_READ|PROT_WRITE.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::MutableAnonymousCommit(size_t capacity)
{
    // Returned value is in KB so convert to bytes.
    if (m_Data == nullptr) {
        assert(UsingAnonymous());
        if (m_OverrideViewSizeBytes > 0) {
            MutableReserve(m_OverrideViewSizeBytes);
        } else {
            MutableReserve(m_PhysicalMemoryCapacityKB * 1024);
        }
    }

    if (capacity > m_CapacityInElts) {
        const auto bytesCount = capacity * sizeof(EltT);

        // Commit by changing protection from PROT_NONE to PROT_READ|PROT_WRITE.
        if (mprotect(m_Data, bytesCount, PROT_READ | PROT_WRITE) != 0) {
            auto code = errno;
            char buf[256];
            snprintf(buf, sizeof(buf), "Failed to commit memory: %d", code);
            throw FractalSharkSeriousException(buf, false);
        }

        GlobalCallstacks->LogAllocCallstack(bytesCount, m_Data);
        m_CapacityInElts = capacity;
    }
}

// =========================================================================
// MutableReserve — reserve address space with PROT_NONE via mmap.
// =========================================================================

template <class EltT>
void
GrowableVector<EltT>::MutableReserve(size_t new_reserved_bytes)
{
    assert(UsingAnonymous()); // This function is only for anonymous memory.

    void *res = mmap(nullptr, new_reserved_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    GlobalCallstacks->LogReserveCallstack(new_reserved_bytes, res);

    if (res == MAP_FAILED) {
        auto code = errno;
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to reserve memory: %d", code);
        throw FractalSharkSeriousException(buf, false);
    }

    m_Data = static_cast<EltT *>(res);

    // Track the reserved size for later munmap without allocating.
    m_MappedFile = MappingViewState(new_reserved_bytes);
}

// =========================================================================
// Explicit template instantiations — must match Win32 Vectors.cpp exactly.
// =========================================================================

#define InstantiateLAInfoDeepGrowableVector(IterType, T, SubType, PExtras)                              \
    template class GrowableVector<LAInfoDeep<IterType, T, SubType, PExtras>>

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Disable);

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<double>, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Bad);

InstantiateLAInfoDeepGrowableVector(uint32_t, float, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, double, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    HDRFloat<double>,
                                    double,
                                    PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint32_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::SimpleCompression);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Disable);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<double>, double, PerturbExtras::Bad);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::Bad);

InstantiateLAInfoDeepGrowableVector(uint64_t, float, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, double, double, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    CudaDblflt<MattDblflt>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    HDRFloat<double>,
                                    double,
                                    PerturbExtras::SimpleCompression);
InstantiateLAInfoDeepGrowableVector(uint64_t,
                                    HDRFloat<CudaDblflt<MattDblflt>>,
                                    CudaDblflt<MattDblflt>,
                                    PerturbExtras::SimpleCompression);

#define InstantiateGrowableVector(EltT) template class GrowableVector<EltT>

InstantiateGrowableVector(LAStageInfo<uint32_t>);
InstantiateGrowableVector(LAStageInfo<uint64_t>);
InstantiateGrowableVector(uint8_t);
InstantiateGrowableVector(uint16_t);
InstantiateGrowableVector(uint32_t);
InstantiateGrowableVector(uint64_t);

#define InstantiateGPUReferenceIterGrowableVector(T, PExtras)                                           \
    template class GrowableVector<GPUReferenceIter<T, PExtras>>

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
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>,
                                          PerturbExtras::SimpleCompression);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::MaxCompression);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>,
                                          PerturbExtras::MaxCompression);

InstantiateGPUReferenceIterGrowableVector(float, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(double, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<float>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<double>, PerturbExtras::Bad);
InstantiateGPUReferenceIterGrowableVector(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);

#endif // _WIN32
