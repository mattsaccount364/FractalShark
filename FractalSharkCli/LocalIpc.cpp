#include "stdafx.h"

#include "LocalIpc.h"

#include <array>
#include <utility>

namespace FractalSharkCli {
namespace {

constexpr uint32_t ProtocolMagic = 0x50495346U; // Four little-endian bytes: FSIP.
constexpr uint32_t MaximumArgumentCount = 256;
constexpr size_t MaximumArgumentBytes = 1024 * 1024;
constexpr uint32_t MaximumBatchItems = 4096;
constexpr size_t MaximumBatchBytes = 64 * 1024 * 1024;
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

void
AppendUint64(std::vector<uint8_t> &buffer, uint64_t value)
{
    AppendUint32(buffer, static_cast<uint32_t>(value));
    AppendUint32(buffer, static_cast<uint32_t>(value >> 32U));
}

uint32_t
ReadUint32(const uint8_t *bytes)
{
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

uint64_t
ReadUint64(const uint8_t *bytes)
{
    return static_cast<uint64_t>(ReadUint32(bytes)) |
           (static_cast<uint64_t>(ReadUint32(bytes + 4)) << 32U);
}

bool
AppendArguments(std::vector<uint8_t> &bytes,
                const std::vector<std::string> &arguments,
                size_t maximumBytes,
                std::string &error)
{
    if (arguments.size() > MaximumArgumentCount) {
        error = "too many request arguments";
        return false;
    }
    if (bytes.size() > maximumBytes - sizeof(uint32_t)) {
        error = "request arguments are too large";
        return false;
    }
    AppendUint32(bytes, static_cast<uint32_t>(arguments.size()));
    size_t argumentBytes = 0;
    for (const auto &argument : arguments) {
        if (argument.size() > MaximumArgumentBytes ||
            argumentBytes > MaximumArgumentBytes - argument.size() ||
            bytes.size() > maximumBytes - sizeof(uint32_t) - argument.size()) {
            error = "request arguments are too large";
            return false;
        }
        argumentBytes += argument.size();
        AppendUint32(bytes, static_cast<uint32_t>(argument.size()));
        bytes.insert(bytes.end(), argument.begin(), argument.end());
    }
    return true;
}

std::vector<uint8_t>
BuildRequestBytes(const IpcRequest &request, std::string &error)
{
    std::vector<uint8_t> bytes;
    AppendUint32(bytes, ProtocolMagic);
    AppendUint32(bytes, static_cast<uint32_t>(request.Operation));
    if (request.Operation == IpcOperation::Batch) {
        if (request.BatchArguments.empty() || request.BatchArguments.size() > MaximumBatchItems) {
            error = "batch must contain 1 to 4096 images";
            return {};
        }
        AppendUint32(bytes, static_cast<uint32_t>(request.BatchArguments.size()));
        for (const auto &arguments : request.BatchArguments) {
            if (!AppendArguments(bytes, arguments, MaximumBatchBytes, error)) {
                return {};
            }
        }
    } else if (!AppendArguments(bytes,
                                request.Arguments,
                                MaximumArgumentBytes + MaximumArgumentCount * 4 + 16,
                                error)) {
        return {};
    }
    return bytes;
}

bool
ReadArguments(Environment::LocalIpcConnection &connection,
              uint32_t argumentCount,
              std::vector<std::string> &arguments,
              size_t &totalBytes,
              size_t maximumBytes,
              std::string &error)
{
    if (totalBytes > maximumBytes) {
        error = "IPC request arguments are too large";
        return false;
    }
    if (argumentCount > MaximumArgumentCount) {
        error = "too many IPC request arguments";
        return false;
    }
    arguments.clear();
    arguments.reserve(argumentCount);
    size_t argumentBytes = 0;
    for (uint32_t i = 0; i < argumentCount; i++) {
        std::array<uint8_t, 4> lengthBytes{};
        if (!connection.ReadExact(lengthBytes.data(), lengthBytes.size(), error)) {
            return false;
        }
        const uint32_t length = ReadUint32(lengthBytes.data());
        if (length > MaximumArgumentBytes || argumentBytes > MaximumArgumentBytes - length ||
            totalBytes > maximumBytes - sizeof(uint32_t) - length) {
            error = "IPC request arguments are too large";
            return false;
        }
        std::string argument(length, '\0');
        if (!connection.ReadExact(argument.data(), argument.size(), error)) {
            return false;
        }
        arguments.push_back(std::move(argument));
        argumentBytes += length;
        totalBytes += sizeof(uint32_t) + length;
    }
    return true;
}

bool
ReadHeader(Environment::LocalIpcConnection &connection, uint8_t *header, size_t size, std::string &error)
{
    if (!connection.ReadExact(header, size, error)) {
        return false;
    }
    if (ReadUint32(header) != ProtocolMagic) {
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

    std::array<uint8_t, 16> header{};
    if (!ReadHeader(connection, header.data(), header.size(), error)) {
        return false;
    }

    const uint32_t stdoutLength = ReadUint32(header.data() + 8);
    const uint32_t stderrLength = ReadUint32(header.data() + 12);
    if (stdoutLength > MaximumResponseBytes || stderrLength > MaximumResponseBytes ||
        static_cast<size_t>(stdoutLength) > MaximumResponseBytes - stderrLength) {
        error = "IPC response is too large";
        return false;
    }

    response.Status = static_cast<int32_t>(ReadUint32(header.data() + 4));
    response.Stdout.resize(stdoutLength);
    response.Stderr.resize(stderrLength);
    return connection.ReadExact(response.Stdout.data(), response.Stdout.size(), error) &&
           connection.ReadExact(response.Stderr.data(), response.Stderr.size(), error);
}

bool
SendBatch(std::string_view endpoint,
          const IpcRequest &request,
          const BatchResultHandler &onResult,
          size_t &completedCount,
          std::string &error)
{
    completedCount = 0;
    if (request.Operation != IpcOperation::Batch) {
        error = "SendBatch requires a batch request";
        return false;
    }
    const std::vector<uint8_t> requestBytes = BuildRequestBytes(request, error);
    if (requestBytes.empty()) {
        return false;
    }
    Environment::LocalIpcConnection connection =
        Environment::ConnectLocalIpc(ServiceName, endpoint, ConnectionTimeoutMilliseconds, error);
    if (!connection.IsOpen() ||
        !connection.WriteExact(requestBytes.data(), requestBytes.size(), error)) {
        return false;
    }

    for (size_t index = 0; index < request.BatchArguments.size(); ++index) {
        std::array<uint8_t, 64> header{};
        if (!ReadHeader(connection, header.data(), header.size(), error)) {
            if (index == 0) {
                error += "; server may not support batch requests";
            }
            return false;
        }
        const uint32_t stdoutLength = ReadUint32(header.data() + 8);
        const uint32_t stderrLength = ReadUint32(header.data() + 12);
        if (stdoutLength > MaximumResponseBytes || stderrLength > MaximumResponseBytes ||
            static_cast<size_t>(stdoutLength) > MaximumResponseBytes - stderrLength) {
            error = "IPC batch response is too large";
            return false;
        }
        IpcBatchResult result;
        result.Response.Status = static_cast<int32_t>(ReadUint32(header.data() + 4));
        result.CliImageMs = ReadUint64(header.data() + 16);
        result.OverallMs = ReadUint64(header.data() + 24);
        result.PerPixelMs = ReadUint64(header.data() + 32);
        result.RefOrbitMs = ReadUint64(header.data() + 40);
        result.Width = ReadUint32(header.data() + 48);
        result.Height = ReadUint32(header.data() + 52);
        result.Iterations = ReadUint64(header.data() + 56);
        result.Response.Stdout.resize(stdoutLength);
        result.Response.Stderr.resize(stderrLength);
        if (!connection.ReadExact(result.Response.Stdout.data(), stdoutLength, error) ||
            !connection.ReadExact(result.Response.Stderr.data(), stderrLength, error)) {
            return false;
        }
        onResult(index, result);
        ++completedCount;
    }
    return true;
}

bool
ReadRequest(Environment::LocalIpcConnection &connection, IpcRequest &request, std::string &error)
{
    std::array<uint8_t, 12> header{};
    if (!ReadHeader(connection, header.data(), header.size(), error)) {
        return false;
    }

    const uint32_t operation = ReadUint32(header.data() + 4);
    if (operation != static_cast<uint32_t>(IpcOperation::Render) &&
        operation != static_cast<uint32_t>(IpcOperation::Shutdown) &&
        operation != static_cast<uint32_t>(IpcOperation::Batch)) {
        error = "invalid FractalSharkCli IPC operation";
        return false;
    }
    const uint32_t argumentCount = ReadUint32(header.data() + 8);
    request.Operation = static_cast<IpcOperation>(operation);
    request.Arguments.clear();
    request.BatchArguments.clear();
    if (request.Operation == IpcOperation::Batch) {
        if (argumentCount == 0 || argumentCount > MaximumBatchItems) {
            error = "invalid IPC batch image count";
            return false;
        }
        request.BatchArguments.reserve(argumentCount);
        size_t totalBytes = header.size();
        for (uint32_t index = 0; index < argumentCount; ++index) {
            std::array<uint8_t, 4> countBytes{};
            if (!connection.ReadExact(countBytes.data(), countBytes.size(), error)) {
                return false;
            }
            totalBytes += countBytes.size();
            request.BatchArguments.emplace_back();
            if (!ReadArguments(connection,
                               ReadUint32(countBytes.data()),
                               request.BatchArguments.back(),
                               totalBytes,
                               MaximumBatchBytes,
                               error)) {
                return false;
            }
        }
        return true;
    }
    size_t totalBytes = header.size();
    return ReadArguments(connection,
                         argumentCount,
                         request.Arguments,
                         totalBytes,
                         MaximumArgumentBytes + MaximumArgumentCount * 4 + 16,
                         error);
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
    header.reserve(16);
    AppendUint32(header, ProtocolMagic);
    AppendUint32(header, static_cast<uint32_t>(response.Status));
    AppendUint32(header, static_cast<uint32_t>(response.Stdout.size()));
    AppendUint32(header, static_cast<uint32_t>(response.Stderr.size()));
    return connection.WriteExact(header.data(), header.size(), error) &&
           connection.WriteExact(response.Stdout.data(), response.Stdout.size(), error) &&
           connection.WriteExact(response.Stderr.data(), response.Stderr.size(), error);
}

bool
WriteBatchResult(Environment::LocalIpcConnection &connection,
                 const IpcBatchResult &result,
                 std::string &error)
{
    const IpcResponse &response = result.Response;
    if (response.Stdout.size() > MaximumResponseBytes || response.Stderr.size() > MaximumResponseBytes ||
        response.Stdout.size() > MaximumResponseBytes - response.Stderr.size()) {
        error = "IPC batch response is too large";
        return false;
    }
    std::vector<uint8_t> header;
    header.reserve(64);
    AppendUint32(header, ProtocolMagic);
    AppendUint32(header, static_cast<uint32_t>(response.Status));
    AppendUint32(header, static_cast<uint32_t>(response.Stdout.size()));
    AppendUint32(header, static_cast<uint32_t>(response.Stderr.size()));
    AppendUint64(header, result.CliImageMs);
    AppendUint64(header, result.OverallMs);
    AppendUint64(header, result.PerPixelMs);
    AppendUint64(header, result.RefOrbitMs);
    AppendUint32(header, result.Width);
    AppendUint32(header, result.Height);
    AppendUint64(header, result.Iterations);
    return connection.WriteExact(header.data(), header.size(), error) &&
           connection.WriteExact(response.Stdout.data(), response.Stdout.size(), error) &&
           connection.WriteExact(response.Stderr.data(), response.Stderr.size(), error);
}

} // namespace FractalSharkCli
