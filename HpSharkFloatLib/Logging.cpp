#include "Logging.h"

#include "ConsoleLog.h"

#include <cuda_runtime_api.h>

#include <sstream>

namespace HpShark {

void
LogCudaError(uint32_t errorCode, const char *file, int line)
{
    LogCudaError({}, errorCode, file, line);
}

void
LogCudaError(std::string_view context, uint32_t errorCode, const char *file, int line)
{
    const char *errorMessage = cudaGetErrorString(static_cast<cudaError_t>(errorCode));

    std::ostringstream message;
    if (!context.empty()) {
        message << context << ": ";
    }
    message << "Error from CUDA: code " << errorCode << ". Message: \""
            << (errorMessage == nullptr ? "<unknown>" : errorMessage) << '"';

    FractalSharkLog::Write(message.str(), file, line);
}

} // namespace HpShark
