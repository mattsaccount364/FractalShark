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
    LAStageInfo
};

std::wstring GetFileExtension(GrowableVectorTypes Type);

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

public:
    EltT* GetData() const;

    std::wstring GetFilename() const;

    size_t GetCapacity() const;

    size_t GetSize() const;

    bool ValidFile() const;

    GrowableVector(const GrowableVector& other) = delete;
    GrowableVector& operator=(const GrowableVector& other) = delete;
    GrowableVector(GrowableVector&& other);

    GrowableVector& operator=(GrowableVector&& other);

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

    ~GrowableVector();

    EltT& operator[](size_t index);

    const EltT& operator[](size_t index) const;

    void PushBack(const EltT& val);

    void PopBack();

    EltT& Back();

    const EltT& Back() const;

    void Clear();

    void CloseMapping(bool CloseFileToo);

    void Trim();

    AddPointOptions GetAddPointOptions() const;

    void MutableReserveKeepFileSize(size_t capacity);
    void MutableResize(size_t capacity, size_t size);
    void MutableResize(size_t size);

private:
    static constexpr auto GrowByElts = 512 * 1024;

    bool UsingAnonymous() const;

    void MutableFileCommit(size_t new_elt_count);

    void MutableAnonymousCommit(size_t new_elt_count);

    void MutableReserve(size_t new_reserved_bytes);
};