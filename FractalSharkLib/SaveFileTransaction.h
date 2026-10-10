#pragma once

#include "SaveResult.h"
#include <functional>
#include <ostream>

// The writer receives an exclusively created temporary file beside the destination.
// Only a successful write and close can publish it. Destruction removes unfinished
// temporary files; readers see either the old complete file or the new complete file.
class SaveFileTransaction {
public:
    static SaveResult Write(const std::wstring &destination,
                            SaveOverwritePolicy overwrite,
                            const std::function<void(std::ostream &)> &writer);
};
