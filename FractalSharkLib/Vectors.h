#pragma once

#include "HighPrecision.h"

enum class AddPointOptions {
    DontSave,
    EnableWithSave,
    EnableWithoutSave,
    OpenExistingWithSave,
};

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAInfoDeep;

template<typename IterType>
class LAStageInfo;

enum class GrowableVectorTypes {
    Metadata,
    GPUReferenceIter,
    LAInfoDeep,
    LAStageInfo,
    DebugOutput
};

std::wstring GetFileExtension(GrowableVectorTypes Type);

void VectorStaticInit();

// The purpose of this class is to manage a memory-mapped file, using win32 APIs
// such as MapViewOfFile, or to provide non-file-backed storage that's virtually contiguous.
// This is used to load and save the orbit data, when using a file.
// If no filename is provided, it's anonymous memory.
// The vector is growable, and if it is backed by a file to hold the data,
// it uses memory mapping. The vector is resizable and is not thread safe.
template<class EltT>
class GrowableVector {
private:
    using Handle = void*;

    Handle m_FileHandle;
    Handle m_MappedFile;
    size_t m_UsedSizeInElts;
    size_t m_CapacityInElts;
    EltT* m_Data;
    AddPointOptions m_AddPointOptions;

    std::wstring m_Filename;
    size_t m_PhysicalMemoryCapacityKB;
    size_t m_GrowByElts;

public:
    GrowableVector(const GrowableVector& other) = delete;
    GrowableVector& operator=(const GrowableVector& other) = delete;
    GrowableVector(GrowableVector&& other) noexcept;

    GrowableVector& operator=(GrowableVector&& other) noexcept;

    // The default constructor creates an empty vector.
    GrowableVector();

    // The constructor takes the file to open or create
    // It maps enough memory to accomodate the provided orbit size.
    GrowableVector(AddPointOptions add_point_options, std::wstring filename);

    // This one takes a filename and size and uses the file specified
    // to back the vector.
    GrowableVector(
        AddPointOptions add_point_options,
        std::wstring filename,
        size_t initial_size);

    // The destructor closes the file and cleans up the memory.
    ~GrowableVector();

    // The subscript operator allows access to the elements of the vector.
    EltT& operator[](size_t index);

    // Constant version of the subscript operator.
    const EltT& operator[](size_t index) const;

    // The GetData method returns a pointer to the data.
    EltT* GetData() const;

    // The GetFilename method returns the filename of the file backing the vector.
    std::wstring GetFilename() const;

    // The GetCapacity method returns the capacity of the vector.
    size_t GetCapacity() const;

    // The GetSize method returns the size of the vector.
    size_t GetSize() const;

    // The ValidFile method returns true if the file is valid.
    bool ValidFile() const;

    // The Back method returns the last element of the vector.
    const EltT& Back() const;

    // The PushBack method adds an element to the end of the vector.
    void PushBack(const EltT& val);

    // The PopBack method removes the last element of the vector.
    void PopBack();

    // The Back method returns the last element of the vector.
    EltT& Back();

    // The Clear method clears the vector and resets the size to zero.
    void Clear();

    // The CloseMapping method closes the memory mapping and the file.
    void CloseMapping();

    // The Trim method trims the vector to the size of the data.
    // This is useful to ensure that no extra memory is allocated.
    void Trim();

    // The GetAddPointOptions method returns the AddPointOptions,
    // which specifies whether to save the data to the file,
    // or to keep it in memory only.
    AddPointOptions GetAddPointOptions() const;

    // The MutableReserveKeepFileSize method reserves memory for the vector,
    void MutableReserveKeepFileSize(size_t capacity);

    // The MutableResize method resizes the vector,
    // using the specified capacity and size.
    void MutableResize(size_t capacity, size_t size);

    // The MutableResize method resizes the vector, using
    // the same value for both the capacity and the size.
    void MutableResize(size_t size);

    // The GrowVectorIfNeeded method grows the vector if needed,
    // reserving additional memory in the process.
    void GrowVectorIfNeeded();

private:
    // The InternalOpenFile method opens the file but
    // does not map it to memory.
    uint32_t InternalOpenFile();

    // The MutableFileResizeOpen method extends
    // the section of the file that is mapped to memory.
    void MutableFileResizeOpen(size_t capacity);

    // The TrimEnableWithoutSave trims the vector
    // in the case of a temporary file-backed vector.
    void TrimEnableWithoutSave();

    // The TrimEnableWithSave trims the vector
    // sets the file size to the size of the vector.
    void TrimEnableWithSave();

    // The UsingAnonymous method returns true if the vector
    // is backed by anonymous memory.
    bool UsingAnonymous() const;

    // The MutableFileCommit method creates and maps a section
    // of the file to memory.
    void MutableFileCommit(size_t new_elt_count);

    // The MutableAnonymousCommit method commits regular
    // pagefile backed memory.
    void MutableAnonymousCommit(size_t new_elt_count);

    // The MutableReserve method reserves memory for the vector
    // using anonymous memory.
    void MutableReserve(size_t new_reserved_bytes);
};