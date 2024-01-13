#pragma once

#include "HighPrecision.h"

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAInfoDeep;

template<typename IterType>
class LAStageInfo;

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

    std::wstring Filename;

public:
    EltT* GetData() const;

    std::wstring GetFilename() const;

    size_t GetCapacity() const;

    size_t GetSize() const;

    bool ValidFile() const;

    GrowableVector();

    GrowableVector(const GrowableVector& other) = delete;
    GrowableVector& operator=(const GrowableVector& other) = delete;
    GrowableVector(GrowableVector&& other);

    GrowableVector& operator=(GrowableVector&& other);

    // The constructor takes the file to open or create
    // It maps enough memory to accomodate the provided orbit size.
    GrowableVector(AddPointOptions add_point_options, const std::wstring filename);

    // This one takes a filename and size and uses the file specified
    // to back the vector.
    GrowableVector(
        AddPointOptions add_point_options,
        const std::wstring filename,
        size_t initial_size);

    ~GrowableVector();

    EltT& operator[](size_t index);

    const EltT& operator[](size_t index) const;

    void PushBack(const EltT& val);

    void PopBack();

    EltT& Back();

    const EltT& Back() const;

    void Clear();

    void CloseMapping(bool /*destruct*/);

    void Trim();

    bool FileBacked() const;

    bool MutableCommit(size_t new_elt_count);

private:
    bool UsingAnonymous() const;

    bool MutableFileCommit(size_t new_elt_count);

    bool MutableAnonymousCommit(size_t new_elt_count);

    void MutableReserve(size_t new_reserved_bytes);
};