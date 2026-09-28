#pragma once

#include "LocalIpcTransport.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace FractalSharkCli {

inline constexpr std::string_view ServiceName = "FractalSharkCli";

enum class IpcOperation : uint32_t { Render = 1, Shutdown = 2, Batch = 3 };

struct IpcRequest {
    IpcOperation Operation = IpcOperation::Render;
    std::vector<std::string> Arguments;
    std::vector<std::vector<std::string>> BatchArguments;
};

struct IpcResponse {
    int32_t Status = 1;
    std::string Stdout;
    std::string Stderr;
};

struct IpcBatchResult {
    IpcResponse Response;
    uint64_t CliImageMs = 0;
    uint64_t OverallMs = 0;
    uint64_t PerPixelMs = 0;
    uint64_t RefOrbitMs = 0;
    uint64_t Iterations = 0;
    uint32_t Width = 0;
    uint32_t Height = 0;
};

using BatchResultHandler = std::function<void(size_t, const IpcBatchResult &)>;

bool SendRequest(std::string_view endpoint,
                 const IpcRequest &request,
                 IpcResponse &response,
                 std::string &error);
bool SendBatch(std::string_view endpoint,
               const IpcRequest &request,
               const BatchResultHandler &onResult,
               size_t &completedCount,
               std::string &error);
bool ReadRequest(Environment::LocalIpcConnection &connection, IpcRequest &request, std::string &error);
bool WriteResponse(Environment::LocalIpcConnection &connection,
                   const IpcResponse &response,
                   std::string &error);
bool WriteBatchResult(Environment::LocalIpcConnection &connection,
                      const IpcBatchResult &result,
                      std::string &error);

} // namespace FractalSharkCli
