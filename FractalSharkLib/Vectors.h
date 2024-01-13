#pragma once

// The purpose of this class is to manage a memory-mapped file, using win32 APIs
// such as MapViewOfFile, or to provide non-file-backed storage that's virtually contiguous.
// This is used to load and save the orbit data, when using a file.
// If no filename is provided, it's anonymous memory.
// The vector is growable, and if it is backed by a file to hold the data,
// it uses memory mapping. The vector is resizable and is not thread safe.
template<class T>
class GrowableVector {
private:
    HANDLE hFile;
    HANDLE hMapFile;
    size_t UsedSizeInElts;
    size_t CapacityInElts;
    T* Mutable;

    std::wstring Filename;

public:
    T* GetData() const {
        return Mutable;
    }

    std::wstring GetFilename() const {
        return Filename;
    }

    size_t GetCapacity() const {
        return CapacityInElts;
    }

    size_t GetSize() const {
        return UsedSizeInElts;
    }

    bool ValidFile() const {
        return Mutable != nullptr;
    }

    GrowableVector() : GrowableVector{ L"" } {
    }

    GrowableVector(const GrowableVector& other) = delete;
    GrowableVector& operator=(const GrowableVector& other) = delete;
    GrowableVector(GrowableVector&& other) :
        hFile{ other.hFile },
        hMapFile{ other.hMapFile },
        UsedSizeInElts{ other.UsedSizeInElts },
        CapacityInElts{ other.CapacityInElts },
        Mutable{ other.Mutable },
        Filename{ other.Filename } {

        other.hFile = nullptr;
        other.hMapFile = nullptr;
        other.UsedSizeInElts = 0;
        other.CapacityInElts = 0;
        other.Mutable = nullptr;
    }

    GrowableVector& operator=(GrowableVector&& other) {
        hFile = other.hFile;
        hMapFile = other.hMapFile;
        UsedSizeInElts = other.UsedSizeInElts;
        CapacityInElts = other.CapacityInElts;
        Mutable = other.Mutable;
        Filename = other.Filename;

        other.hFile = nullptr;
        other.hMapFile = nullptr;
        other.UsedSizeInElts = 0;
        other.CapacityInElts = 0;
        other.Mutable = nullptr;
        other.Filename = {};

        return *this;
    }

    // The constructor takes the file to open or create
    // It maps enough memory to accomodate the provided orbit size.
    GrowableVector(const std::wstring filename)
        : hFile{},
        hMapFile{},
        UsedSizeInElts{},
        CapacityInElts{},
        Mutable{},
        Filename{ filename }
    {
        size_t CapacityInKB;
        auto ret = GetPhysicallyInstalledSystemMemory(&CapacityInKB);
        if (ret == FALSE) {
            ::MessageBox(NULL, L"Failed to get system memory", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        // If the Filename is empty, use virtual memory to back the region.
        // Otherwise, use the provided file.
        if (UsingAnonymous()) {
            // Returned value is in KB so convert to bytes.
            MutableReserve(CapacityInKB * 1024);
        }
        else {
            // 32 MB.  We're not allocating some function of RAM on disk.
            // It should be doing a sparse allocation but who knows.
            MutableFileCommit(1024);
        }
    }

    // This one takes a filename and size and uses the file specified
    // to back the vector.
    GrowableVector(const std::wstring filename, size_t initial_size)
        : GrowableVector{ filename }
    {
        UsedSizeInElts = initial_size;
        CapacityInElts = initial_size;
    }

    ~GrowableVector() {
        CloseMapping(true);

        // TODO Delete the file if it's not anonymous.
        //DeleteFile(Filename.c_str());
    }

    T& operator[](size_t index) {
        return Mutable[index];
    }

    const T& operator[](size_t index) const {
        return Mutable[index];
    }

    void PushBack(const T& val) {
        const auto grow_by_elts = 128 * 1024;

        if (UsedSizeInElts == CapacityInElts) {
            MutableCommit(CapacityInElts + grow_by_elts);
        }

        Mutable[UsedSizeInElts] = val;
        UsedSizeInElts++;
    }

    void PopBack() {
        if (UsedSizeInElts > 0) {
            UsedSizeInElts--;
        }
    }

    T& Back() {
        return Mutable[UsedSizeInElts - 1];
    }

    const T& Back() const {
        return Mutable[UsedSizeInElts - 1];
    }

    void Clear() {
        UsedSizeInElts = 0;
    }

    void CloseMapping(bool /*destruct*/) {
        if (!UsingAnonymous()) {
            if (Mutable != nullptr) {
                UnmapViewOfFile(Mutable);
                Mutable = nullptr;
            }
        }
        else {
            if (Mutable != nullptr) {
                VirtualFree(Mutable, 0, MEM_RELEASE);
                Mutable = nullptr;
            }
        }

        if (hMapFile != nullptr) {
            CloseHandle(hMapFile);
            hMapFile = nullptr;
        }

        if (hFile != nullptr) {
            CloseHandle(hFile);
            hFile = nullptr;
        }

        // UsedSizeInElts is unchanged.
        CapacityInElts = 0;
    }

    void Trim() {
        // TODO No-op for now.
    }

    bool FileBacked() const {
        return !UsingAnonymous();
    }

private:
    bool MutableCommit(size_t new_elt_count) {
        if (!UsingAnonymous()) {
            return MutableFileCommit(new_elt_count);
        }
        else {
            return MutableAnonymousCommit(new_elt_count);
        }
    }

    bool UsingAnonymous() const {
        return Filename == L"";
    }

    bool MutableFileCommit(size_t new_elt_count) {
        if (new_elt_count <= CapacityInElts) {
            return true;
        }

        CloseMapping(false);

        hFile = CreateFile(Filename.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            NULL,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            ::MessageBox(NULL, L"Failed to open file for reading", L"", MB_OK | MB_APPLMODAL);
            return false;
        }

        auto last_error = GetLastError();

        if (last_error == 0 || last_error == ERROR_ALREADY_EXISTS) {
            static_assert(sizeof(size_t) == sizeof(uint64_t), "!");
            static_assert(sizeof(DWORD) == sizeof(uint32_t), "!");
            uint64_t total_new_size = new_elt_count * sizeof(T);
            DWORD high = total_new_size >> 32;
            DWORD low = total_new_size & 0xFFFFFFFF;

            LARGE_INTEGER existing_file_size{};
            if (last_error == ERROR_ALREADY_EXISTS) {
                BOOL ret = GetFileSizeEx(hFile, &existing_file_size);
                if (ret == FALSE) {
                    ::MessageBox(NULL, L"Failed to get file size", L"", MB_OK | MB_APPLMODAL);
                    return false;
                }

                //high = existing_file_size.HighPart;
                //low = existing_file_size.LowPart;

                // Note: by using the file size to find the used size, the implication
                // is that resizing the vector only takes place once the vector is full.
                // This is not the case for anonymous memory.
                // So, MutableCommit is kept private and internal, and used only to grow.
                UsedSizeInElts = existing_file_size.QuadPart / sizeof(T);
            } else {
                UsedSizeInElts = 0;
            }

            CapacityInElts = new_elt_count;

            hMapFile = CreateFileMapping(
                hFile,
                NULL,
                PAGE_READWRITE,
                high,
                low,
                NULL);
            if (hMapFile == nullptr) {
                ::MessageBox(NULL, L"Failed to create file mapping", L"", MB_OK | MB_APPLMODAL);
                return false;
            }

            Mutable = static_cast<T*>(MapViewOfFile(hMapFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0));
            if (Mutable == nullptr) {
                ::MessageBox(NULL, L"Failed to map view of file", L"", MB_OK | MB_APPLMODAL);
                return false;
            }
        }
        else {
            auto ret = GetLastError();
            std::wstring err = L"Failed to open file: ";
            err += std::to_wstring(ret);
            ::MessageBox(NULL, err.c_str(), L"", MB_OK | MB_APPLMODAL);
            return false;
        }

        return true;
    }

    bool MutableAnonymousCommit(size_t new_elt_count) {
        if (new_elt_count > CapacityInElts) {
            // If we're using anonymous memory, we need to commit the memory.
            // Use VirtualAlloc to allocate additional space for the vector.
            // The returned pointer should be the same as the original pointer.

            auto res = VirtualAlloc(
                Mutable,
                new_elt_count * sizeof(T),
                MEM_COMMIT,
                PAGE_READWRITE
            );

            if (Mutable == nullptr) {
                std::wstring err = L"Failed to allocate memory: ";
                auto code = GetLastError();
                err += std::to_wstring(code);

                ::MessageBox(NULL, err.c_str(), L"", MB_OK | MB_APPLMODAL);
                return false;
            }
            else if (Mutable != res) {
                std::wstring err = L"VirtualAlloc returned a different pointer :(";
                err += std::to_wstring(reinterpret_cast<uint64_t>(Mutable));
                err += L" vs ";
                err += std::to_wstring(reinterpret_cast<uint64_t>(res));
                err += L" :(.  Code: ";
                auto code = GetLastError();
                err += std::to_wstring(code);
                ::MessageBox(NULL, err.c_str(), L"", MB_OK | MB_APPLMODAL);
                return false;
            }

            Mutable = static_cast<T*>(res);
            CapacityInElts = new_elt_count;
        }

        return true;
    }

    void MutableReserve(size_t new_reserved_bytes)
    {
        assert(UsingAnonymous());  // This function is only for anonymous memory.

        auto res = VirtualAlloc(
            Mutable,
            new_reserved_bytes,
            MEM_RESERVE,
            PAGE_READWRITE
        );

        if (res == nullptr) {
            std::wstring err = L"Failed to reserve memory: ";
            auto code = GetLastError();
            err += std::to_wstring(code);

            ::MessageBox(NULL, err.c_str(), L"", MB_OK | MB_APPLMODAL);
            return;
        }

        Mutable = static_cast<T*>(res);
    }
};

#if 0
// The purpose of this class is to manage a memory-mapped file, using win32 APIs
// such as MapViewOfFile.  This is used to load and save the orbit data.
template<class T, PerturbExtras PExtras>
class ImmutableMemoryMappedFile {
private:
    HANDLE hFile;
    HANDLE hMapFile;
    size_t OrbitSize;
    const GPUReferenceIter<T, PExtras>* ImmutableFullOrbit;

public:
    const GPUReferenceIter<T, PExtras>* GetImmutableFullOrbit() const {
        return ImmutableFullOrbit;
    }

    size_t GetOrbitSize() const {
        return OrbitSize;
    }

    bool ValidFile() const {
        return ImmutableFullOrbit != nullptr;
    }

    ImmutableMemoryMappedFile() :
        hFile{},
        hMapFile{},
        OrbitSize{},
        ImmutableFullOrbit{} {
    }

    ImmutableMemoryMappedFile(const ImmutableMemoryMappedFile& other) = delete;
    ImmutableMemoryMappedFile& operator=(const ImmutableMemoryMappedFile& other) = delete;
    ImmutableMemoryMappedFile(ImmutableMemoryMappedFile&& other) :
        hFile{ other.hFile },
        hMapFile{ other.hMapFile },
        OrbitSize{ other.OrbitSize },
        ImmutableFullOrbit{ other.ImmutableFullOrbit } {

        other.hFile = nullptr;
        other.hMapFile = nullptr;
        other.OrbitSize = 0;
        other.ImmutableFullOrbit = nullptr;
    }

    ImmutableMemoryMappedFile& operator=(ImmutableMemoryMappedFile&& other) {
        hFile = other.hFile;
        hMapFile = other.hMapFile;
        OrbitSize = other.OrbitSize;
        ImmutableFullOrbit = other.ImmutableFullOrbit;

        other.hFile = nullptr;
        other.hMapFile = nullptr;
        other.OrbitSize = 0;
        other.ImmutableFullOrbit = nullptr;

        return *this;
    }

    ImmutableMemoryMappedFile(
        const std::wstring orbfilename,
        size_t orbit_size) {

        ImmutableFullOrbit = nullptr;

        hFile = CreateFile(orbfilename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            ::MessageBox(NULL, L"Failed to open file for reading", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        hMapFile = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, NULL);
        if (hMapFile == nullptr) {
            ::MessageBox(NULL, L"Failed to create file mapping", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        ImmutableFullOrbit = static_cast<const GPUReferenceIter<T, PExtras>*>(
            MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0));

        if (ImmutableFullOrbit == nullptr) {
            ::MessageBox(NULL, L"Failed to map view of file", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        OrbitSize = orbit_size;
    }

    ~ImmutableMemoryMappedFile() {
        if (ImmutableFullOrbit != nullptr) {
            UnmapViewOfFile(ImmutableFullOrbit);
            ImmutableFullOrbit = nullptr;
        }

        if (hMapFile != nullptr) {
            CloseHandle(hMapFile);
            hMapFile = nullptr;
        }

        if (hFile != nullptr) {
            CloseHandle(hFile);
            hFile = nullptr;
        }

        OrbitSize = 0;
    }
};

#endif