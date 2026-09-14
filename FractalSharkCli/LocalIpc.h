#pragma once

#include "LocalIpcTransport.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace FractalSharkCli {

inline constexpr std::string_view ServiceName = "FractalSharkCli";

enum class IpcOperation : uint32_t { Render = 1, Shutdown = 2 };

struct IpcRequest {
    IpcOperation Operation = IpcOperation::Render;
    std::vector<std::string> Arguments;
};

struct IpcResponse {
    int32_t Status = 1;
    std::string Stdout;
    std::string Stderr;
};

bool SendRequest(std::string_view endpoint,
                 const IpcRequest &request,
                 IpcResponse &response,
                 std::string &error);
bool ReadRequest(Environment::LocalIpcConnection &connection, IpcRequest &request, std::string &error);
bool WriteResponse(Environment::LocalIpcConnection &connection,
                   const IpcResponse &response,
                   std::string &error);

} // namespace FractalSharkCli
