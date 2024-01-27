#include "stdafx.h"
#include "Utilities.h"


namespace Utilities {
    bool FileExists(const wchar_t* filename)
    {
        DWORD dwAttrib = GetFileAttributes(filename);

        return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
            !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
    }
} // namespace Utilities