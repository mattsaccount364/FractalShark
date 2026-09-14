#include "LocalIpcTransport.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <system_error>
#include <thread>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace Environment {
namespace {

bool
IsValidHandle(std::intptr_t handle)
{
    return handle != -1;
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
LowercaseAscii(std::string_view text)
{
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        result.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
    }
    return result;
}

std::string
RuntimeDirectory()
{
    const char *runtimeDirectory = std::getenv("XDG_RUNTIME_DIR");
    if (!runtimeDirectory || runtimeDirectory[0] == '\0') {
        runtimeDirectory = std::getenv("TMPDIR");
    }
    return runtimeDirectory && runtimeDirectory[0] != '\0' ? runtimeDirectory : "/tmp";
}

std::string
NormalizeEndpoint(std::string_view serviceName, std::string_view endpoint)
{
    if (!endpoint.empty() && endpoint.front() == '/') {
        return std::string(endpoint);
    }

    const std::string service = LowercaseAscii(SanitizeName(serviceName));
    if (endpoint.empty()) {
        return RuntimeDirectory() + "/" + service + "-" +
               std::to_string(static_cast<unsigned long long>(getuid())) + ".sock";
    }
    return RuntimeDirectory() + "/" + service + "-" + SanitizeName(endpoint) + ".sock";
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

} // namespace

LocalIpcConnection::LocalIpcConnection(std::intptr_t nativeHandle) : m_NativeHandle(nativeHandle) {}

LocalIpcConnection::~LocalIpcConnection()
{
    Close();
}

LocalIpcConnection::LocalIpcConnection(LocalIpcConnection &&other) noexcept
    : m_NativeHandle(other.m_NativeHandle)
{
    other.m_NativeHandle = -1;
}

LocalIpcConnection &
LocalIpcConnection::operator=(LocalIpcConnection &&other) noexcept
{
    if (this != &other) {
        Close();
        m_NativeHandle = other.m_NativeHandle;
        other.m_NativeHandle = -1;
    }
    return *this;
}

bool
LocalIpcConnection::IsOpen() const
{
    return IsValidHandle(m_NativeHandle);
}

bool
LocalIpcConnection::ReadExact(void *buffer, size_t size, std::string &error)
{
    if (!IsOpen()) {
        error = "IPC connection is closed";
        return false;
    }

    auto *destination = static_cast<uint8_t *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t received =
            recv(static_cast<int>(m_NativeHandle), destination + offset, size - offset, 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = SystemErrorMessage("recv", errno);
            return false;
        }
        if (received == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += static_cast<size_t>(received);
    }
    return true;
}

bool
LocalIpcConnection::WriteExact(const void *buffer, size_t size, std::string &error)
{
    if (!IsOpen()) {
        error = "IPC connection is closed";
        return false;
    }

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
            error = SystemErrorMessage("send", errno);
            return false;
        }
        if (written == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += static_cast<size_t>(written);
    }
    return true;
}

void
LocalIpcConnection::Close()
{
    if (IsOpen()) {
        close(static_cast<int>(m_NativeHandle));
        m_NativeHandle = -1;
    }
}

LocalIpcListener::~LocalIpcListener()
{
    Close();
}

bool
LocalIpcListener::Open(std::string_view serviceName, std::string_view endpoint, std::string &error)
{
    Close();
    m_Endpoint = NormalizeEndpoint(serviceName, endpoint);

    sockaddr_un address{};
    if (!SetSocketAddress(address, m_Endpoint, error)) {
        m_Endpoint.clear();
        return false;
    }

    struct stat existing {};
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
                error = "another " + std::string(serviceName) +
                        " server is already using endpoint " + m_Endpoint;
                m_Endpoint.clear();
                return false;
            }
            if (probeError != ECONNREFUSED && probeError != ENOENT && probeError != ECONNRESET) {
                error = SystemErrorMessage("probe Unix socket", probeError);
                m_Endpoint.clear();
                return false;
            }
        }

        if (unlink(m_Endpoint.c_str()) != 0 && errno != ENOENT) {
            error = SystemErrorMessage("unlink stale Unix socket", errno);
            m_Endpoint.clear();
            return false;
        }
    } else if (errno != ENOENT) {
        error = SystemErrorMessage("inspect Unix socket", errno);
        m_Endpoint.clear();
        return false;
    }

    const int socketHandle = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socketHandle == -1) {
        error = SystemErrorMessage("socket", errno);
        m_Endpoint.clear();
        return false;
    }

    const socklen_t addressLength =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + m_Endpoint.size() + 1);
    if (bind(socketHandle, reinterpret_cast<const sockaddr *>(&address), addressLength) != 0) {
        error = SystemErrorMessage("bind Unix socket", errno);
        close(socketHandle);
        m_Endpoint.clear();
        return false;
    }
    if (chmod(m_Endpoint.c_str(), S_IRUSR | S_IWUSR) != 0) {
        error = SystemErrorMessage("chmod Unix socket", errno);
        close(socketHandle);
        unlink(m_Endpoint.c_str());
        m_Endpoint.clear();
        return false;
    }
    if (listen(socketHandle, 64) != 0) {
        error = SystemErrorMessage("listen Unix socket", errno);
        close(socketHandle);
        unlink(m_Endpoint.c_str());
        m_Endpoint.clear();
        return false;
    }

    m_NativeHandle = socketHandle;
    m_OwnsEndpoint = true;
    return true;
}

LocalIpcConnection
LocalIpcListener::Accept(std::string &error)
{
    if (!IsValidHandle(m_NativeHandle)) {
        error = "IPC listener is closed";
        return {};
    }

    for (;;) {
        const int connection = accept(static_cast<int>(m_NativeHandle), nullptr, nullptr);
        if (connection != -1) {
            return LocalIpcConnection(connection);
        }
        if (errno != EINTR) {
            error = SystemErrorMessage("accept Unix socket", errno);
            return {};
        }
    }
}

void
LocalIpcListener::Close()
{
    if (IsValidHandle(m_NativeHandle)) {
        close(static_cast<int>(m_NativeHandle));
        m_NativeHandle = -1;
    }
    if (m_OwnsEndpoint && !m_Endpoint.empty()) {
        unlink(m_Endpoint.c_str());
    }
    m_OwnsEndpoint = false;
    m_Endpoint.clear();
}

const std::string &
LocalIpcListener::Endpoint() const
{
    return m_Endpoint;
}

LocalIpcConnection
ConnectLocalIpc(std::string_view serviceName,
                std::string_view endpoint,
                uint32_t timeoutMilliseconds,
                std::string &error)
{
    const std::string path = NormalizeEndpoint(serviceName, endpoint);
    sockaddr_un address{};
    if (!SetSocketAddress(address, path, error)) {
        return {};
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMilliseconds);
    for (;;) {
        const int socketHandle = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socketHandle == -1) {
            error = SystemErrorMessage("socket", errno);
            return {};
        }

        const socklen_t addressLength =
            static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
        if (connect(socketHandle, reinterpret_cast<const sockaddr *>(&address), addressLength) == 0) {
            return LocalIpcConnection(socketHandle);
        }

        const int lastError = errno;
        close(socketHandle);
        if (lastError != ENOENT && lastError != ECONNREFUSED && lastError != ECONNRESET) {
            error = SystemErrorMessage("connect", lastError);
            return {};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = SystemErrorMessage("timed out connecting to Unix socket", lastError);
            return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
}

} // namespace Environment
