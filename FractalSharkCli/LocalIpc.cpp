#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include "stdafx.h"

#include "LocalIpc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <system_error>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#else
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace FractalSharkCli {
namespace {

constexpr uint32_t ProtocolMagic = 0x314B5346U; // Four little-endian bytes: FSK1.
constexpr uint32_t ProtocolVersion = 1;
constexpr uint32_t MaximumArgumentCount = 256;
constexpr size_t MaximumArgumentBytes = 1024 * 1024;
constexpr size_t MaximumResponseBytes = 16 * 1024 * 1024;
constexpr std::chrono::milliseconds ConnectionTimeout{30000};

bool
IsValidHandle(std::intptr_t handle)
{
    return handle != -1;
}

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

std::string
SystemErrorMessage(const char *operation, int errorCode)
{
    return std::string(operation) + ": " + std::system_category().message(errorCode);
}

std::string
SanitizeName(std::string_view name)
{
    std::string result;
    result.reserve(name.size());
    for (unsigned char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
            c == '_' || c == '.') {
            result.push_back(static_cast<char>(c));
        } else {
            result.push_back('_');
        }
    }
    return result;
}

std::string
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

    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

bool
ReadHeader(LocalConnection &connection, uint8_t *header, size_t size, std::string &error)
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

#ifdef _WIN32

HANDLE
AsHandle(std::intptr_t value) { return reinterpret_cast<HANDLE>(value); }

std::intptr_t
AsInteger(HANDLE value)
{
    return reinterpret_cast<std::intptr_t>(value);
}

std::string
WindowsErrorMessage(const char *operation)
{
    return SystemErrorMessage(operation, static_cast<int>(GetLastError()));
}

std::string
PipeName(std::string_view endpoint)
{
    std::string normalized = NormalizeEndpoint(endpoint);
    return normalized;
}

LocalConnection
ConnectToEndpoint(std::string_view endpoint, std::string &error)
{
    const std::string pipeName = PipeName(endpoint);
    const auto deadline = std::chrono::steady_clock::now() + ConnectionTimeout;

    for (;;) {
        HANDLE pipe = CreateFileA(
            pipeName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) {
            return LocalConnection(AsInteger(pipe));
        }

        const DWORD lastError = GetLastError();
        if (lastError != ERROR_PIPE_BUSY && lastError != ERROR_FILE_NOT_FOUND) {
            error = SystemErrorMessage("CreateFile(named pipe)", static_cast<int>(lastError));
            return {};
        }

        if (std::chrono::steady_clock::now() >= deadline) {
            error =
                SystemErrorMessage("timed out connecting to named pipe", static_cast<int>(lastError));
            return {};
        }

        if (lastError == ERROR_PIPE_BUSY) {
            WaitNamedPipeA(pipeName.c_str(), 100);
        } else {
            Sleep(25);
        }
    }
}

#else

std::string
PosixErrorMessage(const char *operation, int errorCode)
{
    return SystemErrorMessage(operation, errorCode);
}

bool
SetSocketAddress(sockaddr_un &address, const std::string &path, std::string &error)
{
    if (path.size() >= sizeof(address.sun_path)) {
        error = "Unix socket endpoint path is too long: " + path;
        return false;
    }

    std::memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return true;
}

LocalConnection
ConnectToEndpoint(std::string_view endpoint, std::string &error)
{
    const std::string path = NormalizeEndpoint(endpoint);
    const auto deadline = std::chrono::steady_clock::now() + ConnectionTimeout;

    for (;;) {
        const int socketHandle = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socketHandle == -1) {
            error = PosixErrorMessage("socket", errno);
            return {};
        }

        sockaddr_un address{};
        if (!SetSocketAddress(address, path, error)) {
            close(socketHandle);
            return {};
        }

        const socklen_t addressLength =
            static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
        if (connect(socketHandle, reinterpret_cast<const sockaddr *>(&address), addressLength) == 0) {
            return LocalConnection(socketHandle);
        }

        const int lastError = errno;
        close(socketHandle);
        if (lastError != ENOENT && lastError != ECONNREFUSED && lastError != ECONNRESET) {
            error = PosixErrorMessage("connect", lastError);
            return {};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = PosixErrorMessage("timed out connecting to Unix socket", lastError);
            return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
}

#endif

} // namespace

LocalConnection::LocalConnection(std::intptr_t nativeHandle) : m_NativeHandle(nativeHandle) {}

LocalConnection::~LocalConnection() { Close(); }

LocalConnection::LocalConnection(LocalConnection &&other) noexcept : m_NativeHandle(other.m_NativeHandle)
{
    other.m_NativeHandle = -1;
}

LocalConnection &
LocalConnection::operator=(LocalConnection &&other) noexcept
{
    if (this != &other) {
        Close();
        m_NativeHandle = other.m_NativeHandle;
        other.m_NativeHandle = -1;
    }
    return *this;
}

bool
LocalConnection::IsOpen() const
{
    return IsValidHandle(m_NativeHandle);
}

bool
LocalConnection::ReadExact(void *buffer, size_t size, std::string &error)
{
    if (!IsOpen()) {
        error = "IPC connection is closed";
        return false;
    }

#ifdef _WIN32
    auto *destination = static_cast<uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const DWORD requested = static_cast<DWORD>(std::min<size_t>(size - offset, UINT32_MAX));
        DWORD received = 0;
        if (!ReadFile(AsHandle(m_NativeHandle), destination + offset, requested, &received, nullptr)) {
            error = WindowsErrorMessage("ReadFile(named pipe)");
            return false;
        }
        if (received == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += received;
    }
#else
    auto *destination = static_cast<uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t received =
            recv(static_cast<int>(m_NativeHandle), destination + offset, size - offset, 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = PosixErrorMessage("recv", errno);
            return false;
        }
        if (received == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += static_cast<size_t>(received);
    }
#endif

    return true;
}

bool
LocalConnection::WriteExact(const void *buffer, size_t size, std::string &error)
{
    if (!IsOpen()) {
        error = "IPC connection is closed";
        return false;
    }

#ifdef _WIN32
    const auto *source = static_cast<const uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const DWORD requested = static_cast<DWORD>(std::min<size_t>(size - offset, UINT32_MAX));
        DWORD written = 0;
        if (!WriteFile(AsHandle(m_NativeHandle), source + offset, requested, &written, nullptr)) {
            error = WindowsErrorMessage("WriteFile(named pipe)");
            return false;
        }
        if (written == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += written;
    }
#else
    const auto *source = static_cast<const uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t written =
            send(static_cast<int>(m_NativeHandle), source + offset, size - offset, flags);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = PosixErrorMessage("send", errno);
            return false;
        }
        if (written == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += static_cast<size_t>(written);
    }
#endif

    return true;
}

void
LocalConnection::Close()
{
    if (!IsOpen()) {
        return;
    }

#ifdef _WIN32
    CloseHandle(AsHandle(m_NativeHandle));
#else
    close(static_cast<int>(m_NativeHandle));
#endif
    m_NativeHandle = -1;
}

LocalListener::~LocalListener() { Close(); }

bool
LocalListener::Open(std::string_view endpoint, std::string &error)
{
    Close();
    m_Endpoint = NormalizeEndpoint(endpoint);
    if (m_Endpoint.empty()) {
        error = "IPC endpoint cannot be empty";
        return false;
    }

#ifdef _WIN32
    const std::string mutexName = "Local\\FractalSharkCli-" + SanitizeName(m_Endpoint) + "-lock";
    HANDLE lock = CreateMutexA(nullptr, TRUE, mutexName.c_str());
    if (!lock) {
        error = WindowsErrorMessage("CreateMutex");
        m_Endpoint.clear();
        return false;
    }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        CloseHandle(lock);
        error = "another FractalSharkCli server is already using endpoint " + m_Endpoint;
        m_Endpoint.clear();
        return false;
    }

    m_LockHandle = AsInteger(lock);
    m_OwnsEndpoint = true;
    return true;
#else
    sockaddr_un address{};
    if (!SetSocketAddress(address, m_Endpoint, error)) {
        m_Endpoint.clear();
        return false;
    }

    struct stat existing{};
    if (lstat(m_Endpoint.c_str(), &existing) == 0) {
        if (!S_ISSOCK(existing.st_mode)) {
            error = "IPC endpoint exists and is not a Unix socket: " + m_Endpoint;
            m_Endpoint.clear();
            return false;
        }

        const int probe = socket(AF_UNIX, SOCK_STREAM, 0);
        if (probe != -1) {
            const socklen_t addressLength =
                static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + m_Endpoint.size() + 1);
            const bool active =
                connect(probe, reinterpret_cast<const sockaddr *>(&address), addressLength) == 0;
            const int probeError = errno;
            close(probe);
            if (active) {
                error = "another FractalSharkCli server is already using endpoint " + m_Endpoint;
                m_Endpoint.clear();
                return false;
            }
            if (probeError != ECONNREFUSED && probeError != ENOENT && probeError != ECONNRESET) {
                error = PosixErrorMessage("probe Unix socket", probeError);
                m_Endpoint.clear();
                return false;
            }
        }

        if (unlink(m_Endpoint.c_str()) != 0 && errno != ENOENT) {
            error = PosixErrorMessage("unlink stale Unix socket", errno);
            m_Endpoint.clear();
            return false;
        }
    } else if (errno != ENOENT) {
        error = PosixErrorMessage("inspect Unix socket", errno);
        m_Endpoint.clear();
        return false;
    }

    const int socketHandle = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socketHandle == -1) {
        error = PosixErrorMessage("socket", errno);
        m_Endpoint.clear();
        return false;
    }

    if (bind(socketHandle,
             reinterpret_cast<const sockaddr *>(&address),
             static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + m_Endpoint.size() + 1)) != 0) {
        error = PosixErrorMessage("bind Unix socket", errno);
        close(socketHandle);
        m_Endpoint.clear();
        return false;
    }
    if (chmod(m_Endpoint.c_str(), S_IRUSR | S_IWUSR) != 0) {
        error = PosixErrorMessage("chmod Unix socket", errno);
        close(socketHandle);
        unlink(m_Endpoint.c_str());
        m_Endpoint.clear();
        return false;
    }
    if (listen(socketHandle, 64) != 0) {
        error = PosixErrorMessage("listen Unix socket", errno);
        close(socketHandle);
        unlink(m_Endpoint.c_str());
        m_Endpoint.clear();
        return false;
    }

    m_NativeHandle = socketHandle;
    m_OwnsEndpoint = true;
    return true;
#endif
}

LocalConnection
LocalListener::Accept(std::string &error)
{
#ifdef _WIN32
    if (!IsValidHandle(m_LockHandle)) {
        error = "IPC listener is closed";
        return {};
    }

    HANDLE pipe = CreateNamedPipeA(m_Endpoint.c_str(),
                                   PIPE_ACCESS_DUPLEX,
                                   PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                   PIPE_UNLIMITED_INSTANCES,
                                   1024 * 1024,
                                   1024 * 1024,
                                   0,
                                   nullptr);
    if (pipe == INVALID_HANDLE_VALUE) {
        error = WindowsErrorMessage("CreateNamedPipe");
        return {};
    }

    if (!ConnectNamedPipe(pipe, nullptr)) {
        const DWORD lastError = GetLastError();
        if (lastError != ERROR_PIPE_CONNECTED) {
            error = SystemErrorMessage("ConnectNamedPipe", static_cast<int>(lastError));
            CloseHandle(pipe);
            return {};
        }
    }

    return LocalConnection(AsInteger(pipe));
#else
    if (!IsValidHandle(m_NativeHandle)) {
        error = "IPC listener is closed";
        return {};
    }

    const int connection = accept(static_cast<int>(m_NativeHandle), nullptr, nullptr);
    if (connection == -1) {
        if (errno == EINTR) {
            return Accept(error);
        }
        error = PosixErrorMessage("accept Unix socket", errno);
        return {};
    }
    return LocalConnection(connection);
#endif
}

void
LocalListener::Close()
{
#ifdef _WIN32
    if (IsValidHandle(m_LockHandle)) {
        CloseHandle(AsHandle(m_LockHandle));
        m_LockHandle = -1;
    }
#else
    if (IsValidHandle(m_NativeHandle)) {
        close(static_cast<int>(m_NativeHandle));
        m_NativeHandle = -1;
    }
    if (m_OwnsEndpoint && !m_Endpoint.empty()) {
        unlink(m_Endpoint.c_str());
    }
#endif
    m_OwnsEndpoint = false;
    m_Endpoint.clear();
}

const std::string &
LocalListener::Endpoint() const
{
    return m_Endpoint;
}

std::string
DefaultEndpoint()
{
#ifdef _WIN32
    char userName[256] = {};
    DWORD userNameLength = static_cast<DWORD>(sizeof(userName));
    std::string user = "user";
    if (GetUserNameA(userName, &userNameLength) != 0 && userNameLength > 0) {
        user.assign(userName, userNameLength - 1);
    }
    return "FractalSharkCli-" + SanitizeName(user);
#else
    const char *runtimeDirectory = std::getenv("XDG_RUNTIME_DIR");
    if (!runtimeDirectory || runtimeDirectory[0] == '\0') {
        runtimeDirectory = std::getenv("TMPDIR");
    }
    const std::string directory =
        (runtimeDirectory && runtimeDirectory[0] != '\0') ? runtimeDirectory : "/tmp";
    return directory + "/fractalsharkcli-" + std::to_string(static_cast<unsigned long long>(getuid())) +
           ".sock";
#endif
}

std::string
NormalizeEndpoint(std::string_view endpoint)
{
    if (endpoint.empty()) {
        return NormalizeEndpoint(DefaultEndpoint());
    }

#ifdef _WIN32
    constexpr std::string_view pipePrefix = R"(\\.\pipe\)";
    if (endpoint.starts_with(pipePrefix)) {
        return std::string(endpoint);
    }
    return std::string(pipePrefix) + SanitizeName(endpoint);
#else
    if (endpoint.front() == '/') {
        return std::string(endpoint);
    }
    const char *runtimeDirectory = std::getenv("XDG_RUNTIME_DIR");
    if (!runtimeDirectory || runtimeDirectory[0] == '\0') {
        runtimeDirectory = std::getenv("TMPDIR");
    }
    const std::string directory =
        (runtimeDirectory && runtimeDirectory[0] != '\0') ? runtimeDirectory : "/tmp";
    return directory + "/fractalsharkcli-" + SanitizeName(endpoint) + ".sock";
#endif
}

bool
SendRequest(std::string_view endpoint,
            const IpcRequest &request,
            IpcResponse &response,
            std::string &error)
{
    LocalConnection connection = ConnectToEndpoint(endpoint, error);
    if (!connection.IsOpen()) {
        return false;
    }

    const std::string requestBytes = BuildRequestBytes(request, error);
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
    if (!connection.ReadExact(response.Stdout.data(), response.Stdout.size(), error) ||
        !connection.ReadExact(response.Stderr.data(), response.Stderr.size(), error)) {
        return false;
    }
    return true;
}

bool
ReadRequest(LocalConnection &connection, IpcRequest &request, std::string &error)
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
WriteResponse(LocalConnection &connection, const IpcResponse &response, std::string &error)
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
    if (!connection.WriteExact(header.data(), header.size(), error)) {
        return false;
    }
    if (!connection.WriteExact(response.Stdout.data(), response.Stdout.size(), error)) {
        return false;
    }
    return connection.WriteExact(response.Stderr.data(), response.Stderr.size(), error);
}

} // namespace FractalSharkCli
