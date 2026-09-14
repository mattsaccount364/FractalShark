#include "stdafx.h"

#include "LocalIpc.h"

#include <array>
#include <limits>
#include <utility>

namespace FractalSharkCli {
namespace {

constexpr uint32_t ProtocolMagic = 0x314B5346U; // Four little-endian bytes: FSK1.
constexpr uint32_t ProtocolVersion = 1;
constexpr uint32_t MaximumArgumentCount = 256;
constexpr size_t MaximumArgumentBytes = 1024 * 1024;
constexpr size_t MaximumResponseBytes = 16 * 1024 * 1024;
constexpr uint32_t ConnectionTimeoutMilliseconds = 30000;

void
AppendUint32(std::vector<uint8_t> &buffer, uint32_t value)
{
    buffer.push_back(static_cast<uint8_t>(value & 0xffU));
    buffer.push_back(static_cast<uint8_t>((value >> 8U) & 0xffU));
    buffer.push_back(static_cast<uint8_t>((value >> 16U) & 0xffU));
    buffer.push_back(static_cast<uint8_t>((value >> 24U) & 0xffU));
}

uint32_t
ReadUint32(const uint8_t *bytes)
{
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

std::vector<uint8_t>
BuildRequestBytes(const IpcRequest &request, std::string &error)
{
    if (request.Arguments.size() > MaximumArgumentCount) {
        error = "too many request arguments";
        return {};
    }

    size_t argumentBytes = 0;
    for (const auto &argument : request.Arguments) {
        if (argument.size() > std::numeric_limits<uint32_t>::max() ||
            argument.size() > MaximumArgumentBytes ||
            argumentBytes > MaximumArgumentBytes - argument.size()) {
            error = "request arguments are too large";
            return {};
        }
        argumentBytes += argument.size();
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(16 + request.Arguments.size() * sizeof(uint32_t) + argumentBytes);
    AppendUint32(bytes, ProtocolMagic);
    AppendUint32(bytes, ProtocolVersion);
    AppendUint32(bytes, static_cast<uint32_t>(request.Operation));
    AppendUint32(bytes, static_cast<uint32_t>(request.Arguments.size()));
    for (const auto &argument : request.Arguments) {
        AppendUint32(bytes, static_cast<uint32_t>(argument.size()));
        bytes.insert(bytes.end(), argument.begin(), argument.end());
    }
    return bytes;
}

bool
ReadHeader(Environment::LocalIpcConnection &connection, uint8_t *header, size_t size, std::string &error)
{
    if (!connection.ReadExact(header, size, error)) {
        return false;
    }
    if (ReadUint32(header) != ProtocolMagic || ReadUint32(header + 4) != ProtocolVersion) {
        error = "invalid FractalSharkCli IPC header";
        return false;
    }
    return true;
}

} // namespace

bool
SendRequest(std::string_view endpoint,
            const IpcRequest &request,
            IpcResponse &response,
            std::string &error)
{
    Environment::LocalIpcConnection connection =
        Environment::ConnectLocalIpc(ServiceName, endpoint, ConnectionTimeoutMilliseconds, error);
    if (!connection.IsOpen()) {
        return false;
    }

    const std::vector<uint8_t> requestBytes = BuildRequestBytes(request, error);
    if (requestBytes.empty() && !error.empty()) {
        return false;
    }
    if (!connection.WriteExact(requestBytes.data(), requestBytes.size(), error)) {
        return false;
    }

    std::array<uint8_t, 20> header{};
    if (!ReadHeader(connection, header.data(), header.size(), error)) {
        return false;
    }

    const uint32_t stdoutLength = ReadUint32(header.data() + 12);
    const uint32_t stderrLength = ReadUint32(header.data() + 16);
    if (stdoutLength > MaximumResponseBytes || stderrLength > MaximumResponseBytes ||
        static_cast<size_t>(stdoutLength) > MaximumResponseBytes - stderrLength) {
        error = "IPC response is too large";
        return false;
    }

    response.Status = static_cast<int32_t>(ReadUint32(header.data() + 8));
    response.Stdout.resize(stdoutLength);
    response.Stderr.resize(stderrLength);
    return connection.ReadExact(response.Stdout.data(), response.Stdout.size(), error) &&
           connection.ReadExact(response.Stderr.data(), response.Stderr.size(), error);
}

bool
ReadRequest(Environment::LocalIpcConnection &connection, IpcRequest &request, std::string &error)
{
    std::array<uint8_t, 16> header{};
    if (!ReadHeader(connection, header.data(), header.size(), error)) {
        return false;
    }

    const uint32_t operation = ReadUint32(header.data() + 8);
    if (operation != static_cast<uint32_t>(IpcOperation::Render) &&
        operation != static_cast<uint32_t>(IpcOperation::Shutdown)) {
        error = "invalid FractalSharkCli IPC operation";
        return false;
    }
    const uint32_t argumentCount = ReadUint32(header.data() + 12);
    if (argumentCount > MaximumArgumentCount) {
        error = "too many IPC request arguments";
        return false;
    }

    request.Operation = static_cast<IpcOperation>(operation);
    request.Arguments.clear();
    request.Arguments.reserve(argumentCount);
    size_t argumentBytes = 0;
    for (uint32_t i = 0; i < argumentCount; i++) {
        std::array<uint8_t, 4> lengthBytes{};
        if (!connection.ReadExact(lengthBytes.data(), lengthBytes.size(), error)) {
            return false;
        }
        const uint32_t length = ReadUint32(lengthBytes.data());
        if (length > MaximumArgumentBytes || argumentBytes > MaximumArgumentBytes - length) {
            error = "IPC request arguments are too large";
            return false;
        }

        std::string argument(length, '\0');
        if (!connection.ReadExact(argument.data(), argument.size(), error)) {
            return false;
        }
        request.Arguments.push_back(std::move(argument));
        argumentBytes += length;
    }
    return true;
}

bool
WriteResponse(Environment::LocalIpcConnection &connection,
              const IpcResponse &response,
              std::string &error)
{
    if (response.Stdout.size() > MaximumResponseBytes || response.Stderr.size() > MaximumResponseBytes ||
        response.Stdout.size() > MaximumResponseBytes - response.Stderr.size() ||
        response.Stdout.size() > std::numeric_limits<uint32_t>::max() ||
        response.Stderr.size() > std::numeric_limits<uint32_t>::max()) {
        error = "IPC response is too large";
        return false;
    }

    std::vector<uint8_t> header;
    header.reserve(20);
    AppendUint32(header, ProtocolMagic);
    AppendUint32(header, ProtocolVersion);
    AppendUint32(header, static_cast<uint32_t>(response.Status));
    AppendUint32(header, static_cast<uint32_t>(response.Stdout.size()));
    AppendUint32(header, static_cast<uint32_t>(response.Stderr.size()));
    return connection.WriteExact(header.data(), header.size(), error) &&
           connection.WriteExact(response.Stdout.data(), response.Stdout.size(), error) &&
           connection.WriteExact(response.Stderr.data(), response.Stderr.size(), error);
}

} // namespace FractalSharkCli
