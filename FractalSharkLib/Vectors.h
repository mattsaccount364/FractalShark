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
    HANDLE m_FileHandle;
    HANDLE m_MappedFile;
    size_t m_UsedSizeInElts;
    size_t m_CapacityInElts;
    T* m_Data;
    RefOrbitCalc::AddPointOptions m_AddPointOptions;

    std::wstring Filename;

public:
    T* GetData() const {
        return m_Data;
    }

    std::wstring GetFilename() const {
        return Filename;
    }

    size_t GetCapacity() const {
        return m_CapacityInElts;
    }

    size_t GetSize() const {
        return m_UsedSizeInElts;
    }

    bool ValidFile() const {
        return m_Data != nullptr;
    }

    GrowableVector() : GrowableVector{ RefOrbitCalc::AddPointOptions::DontSave, L"" } {
    }

    GrowableVector(const GrowableVector& other) = delete;
    GrowableVector& operator=(const GrowableVector& other) = delete;
    GrowableVector(GrowableVector&& other) :
        m_FileHandle{ other.m_FileHandle },
        m_MappedFile{ other.m_MappedFile },
        m_UsedSizeInElts{ other.m_UsedSizeInElts },
        m_CapacityInElts{ other.m_CapacityInElts },
        m_Data{ other.m_Data },
        m_AddPointOptions{ other.m_AddPointOptions },
        Filename{ other.Filename } {

        other.m_FileHandle = nullptr;
        other.m_MappedFile = nullptr;
        other.m_UsedSizeInElts = 0;
        other.m_CapacityInElts = 0;
        other.m_AddPointOptions = RefOrbitCalc::AddPointOptions::DontSave;
        other.m_Data = nullptr;
    }

    GrowableVector& operator=(GrowableVector&& other) {
        m_FileHandle = other.m_FileHandle;
        m_MappedFile = other.m_MappedFile;
        m_UsedSizeInElts = other.m_UsedSizeInElts;
        m_CapacityInElts = other.m_CapacityInElts;
        m_Data = other.m_Data;
        m_AddPointOptions = other.m_AddPointOptions;
        Filename = other.Filename;

        other.m_FileHandle = nullptr;
        other.m_MappedFile = nullptr;
        other.m_UsedSizeInElts = 0;
        other.m_CapacityInElts = 0;
        other.m_Data = nullptr;
        other.m_AddPointOptions = RefOrbitCalc::AddPointOptions::DontSave;
        other.Filename = {};

        return *this;
    }

    // The constructor takes the file to open or create
    // It maps enough memory to accomodate the provided orbit size.
    GrowableVector(RefOrbitCalc::AddPointOptions add_point_options, const std::wstring filename)
        : m_FileHandle{},
        m_MappedFile{},
        m_UsedSizeInElts{},
        m_CapacityInElts{},
        m_Data{},
        m_AddPointOptions{ add_point_options },
        Filename{ filename }
    {
        size_t CapacityInKB;
        auto ret = GetPhysicallyInstalledSystemMemory(&CapacityInKB);
        if (ret == FALSE) {
            ::MessageBox(nullptr, L"Failed to get system memory", L"", MB_OK | MB_APPLMODAL);
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
    GrowableVector(
        RefOrbitCalc::AddPointOptions add_point_options,
        const std::wstring filename,
        size_t initial_size)
        : GrowableVector{ add_point_options, filename }
    {
        m_UsedSizeInElts = initial_size;
        m_CapacityInElts = initial_size;
    }

    ~GrowableVector() {
        CloseMapping(true);

        // TODO Delete the file if it's not anonymous.
        if (m_AddPointOptions == RefOrbitCalc::AddPointOptions::EnableWithoutSave) {
            auto result = DeleteFile(Filename.c_str());
            if (result == FALSE) {
                auto err = GetLastError();
                std::wstring msg = L"Failed to delete file: ";
                msg += std::to_wstring(err);
                ::MessageBox(nullptr, msg.c_str(), L"", MB_OK | MB_APPLMODAL);
            }
        }
    }

    T& operator[](size_t index) {
        return m_Data[index];
    }

    const T& operator[](size_t index) const {
        return m_Data[index];
    }

    void PushBack(const T& val) {
        const auto grow_by_elts = 128 * 1024;

        if (m_UsedSizeInElts == m_CapacityInElts) {
            MutableCommit(m_CapacityInElts + grow_by_elts);
        }

        m_Data[m_UsedSizeInElts] = val;
        m_UsedSizeInElts++;
    }

    void PopBack() {
        if (m_UsedSizeInElts > 0) {
            m_UsedSizeInElts--;
        }
    }

    T& Back() {
        return m_Data[m_UsedSizeInElts - 1];
    }

    const T& Back() const {
        return m_Data[m_UsedSizeInElts - 1];
    }

    void Clear() {
        m_UsedSizeInElts = 0;
    }

    void CloseMapping(bool /*destruct*/) {
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

        if (m_FileHandle != nullptr) {
            CloseHandle(m_FileHandle);
            m_FileHandle = nullptr;
        }

        // m_UsedSizeInElts is unchanged.
        m_CapacityInElts = 0;
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
        if (new_elt_count <= m_CapacityInElts) {
            return true;
        }

        CloseMapping(false);

        m_FileHandle = CreateFile(Filename.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ,
            nullptr,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (m_FileHandle == INVALID_HANDLE_VALUE) {
            ::MessageBox(nullptr, L"Failed to open file for reading", L"", MB_OK | MB_APPLMODAL);
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
                BOOL ret = GetFileSizeEx(m_FileHandle, &existing_file_size);
                if (ret == FALSE) {
                    ::MessageBox(nullptr, L"Failed to get file size", L"", MB_OK | MB_APPLMODAL);
                    return false;
                }

                //high = existing_file_size.HighPart;
                //low = existing_file_size.LowPart;

                // Note: by using the file size to find the used size, the implication
                // is that resizing the vector only takes place once the vector is full.
                // This is not the case for anonymous memory.
                // So, MutableCommit is kept private and internal, and used only to grow.
                m_UsedSizeInElts = existing_file_size.QuadPart / sizeof(T);
            } else {
                m_UsedSizeInElts = 0;
            }

            m_CapacityInElts = new_elt_count;

            m_MappedFile = CreateFileMapping(
                m_FileHandle,
                nullptr,
                PAGE_READWRITE,
                high,
                low,
                nullptr);
            if (m_MappedFile == nullptr) {
                ::MessageBox(nullptr, L"Failed to create file mapping", L"", MB_OK | MB_APPLMODAL);
                return false;
            }

            m_Data = static_cast<T*>(MapViewOfFile(m_MappedFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0));
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

    bool MutableAnonymousCommit(size_t new_elt_count) {
        if (new_elt_count > m_CapacityInElts) {
            // If we're using anonymous memory, we need to commit the memory.
            // Use VirtualAlloc to allocate additional space for the vector.
            // The returned pointer should be the same as the original pointer.

            auto res = VirtualAlloc(
                m_Data,
                new_elt_count * sizeof(T),
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

            m_Data = static_cast<T*>(res);
            m_CapacityInElts = new_elt_count;
        }

        return true;
    }

    void MutableReserve(size_t new_reserved_bytes)
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

        m_Data = static_cast<T*>(res);
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
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            ::MessageBox(nullptr, L"Failed to open file for reading", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        hMapFile = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (hMapFile == nullptr) {
            ::MessageBox(nullptr, L"Failed to create file mapping", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        ImmutableFullOrbit = static_cast<const GPUReferenceIter<T, PExtras>*>(
            MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0));

        if (ImmutableFullOrbit == nullptr) {
            ::MessageBox(nullptr, L"Failed to map view of file", L"", MB_OK | MB_APPLMODAL);
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