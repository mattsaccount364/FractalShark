#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "LocalIpcTransport.h"

#include <algorithm>
#include <chrono>
#include <system_error>

#include <Windows.h>

namespace Environment {
namespace {

bool
IsValidHandle(std::intptr_t handle)
{
    return handle != -1;
}

HANDLE
AsHandle(std::intptr_t value)
{
    return reinterpret_cast<HANDLE>(value);
}

std::intptr_t
AsInteger(HANDLE value)
{
    return reinterpret_cast<std::intptr_t>(value);
}

std::string
SystemErrorMessage(const char *operation, DWORD errorCode)
{
    return std::string(operation) + ": " +
           std::system_category().message(static_cast<int>(errorCode));
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
NormalizeEndpoint(std::string_view serviceName, std::string_view endpoint)
{
    constexpr std::string_view pipePrefix = R"(\\.\pipe\)";
    if (endpoint.starts_with(pipePrefix)) {
        return std::string(endpoint);
    }

    std::string endpointName;
    if (endpoint.empty()) {
        char userName[256] = {};
        DWORD userNameLength = static_cast<DWORD>(sizeof(userName));
        std::string user = "user";
        if (GetUserNameA(userName, &userNameLength) != 0 && userNameLength > 0) {
            user.assign(userName, userNameLength - 1);
        }
        endpointName = SanitizeName(serviceName) + "-" + SanitizeName(user);
    } else {
        endpointName = SanitizeName(endpoint);
    }
    return std::string(pipePrefix) + endpointName;
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
        const DWORD requested = static_cast<DWORD>(std::min<size_t>(size - offset, UINT32_MAX));
        DWORD received = 0;
        if (!ReadFile(AsHandle(m_NativeHandle), destination + offset, requested, &received, nullptr)) {
            error = SystemErrorMessage("ReadFile(named pipe)", GetLastError());
            return false;
        }
        if (received == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += received;
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
        const DWORD requested = static_cast<DWORD>(std::min<size_t>(size - offset, UINT32_MAX));
        DWORD written = 0;
        if (!WriteFile(AsHandle(m_NativeHandle), source + offset, requested, &written, nullptr)) {
            error = SystemErrorMessage("WriteFile(named pipe)", GetLastError());
            return false;
        }
        if (written == 0) {
            error = "IPC peer closed the connection";
            return false;
        }
        offset += written;
    }
    return true;
}

void
LocalIpcConnection::Close()
{
    if (IsOpen()) {
        CloseHandle(AsHandle(m_NativeHandle));
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

    const std::string mutexName =
        "Local\\" + SanitizeName(serviceName) + "-" + SanitizeName(m_Endpoint) + "-lock";
    HANDLE lock = CreateMutexA(nullptr, TRUE, mutexName.c_str());
    if (!lock) {
        error = SystemErrorMessage("CreateMutex", GetLastError());
        m_Endpoint.clear();
        return false;
    }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        CloseHandle(lock);
        error = "another " + std::string(serviceName) + " server is already using endpoint " +
                m_Endpoint;
        m_Endpoint.clear();
        return false;
    }

    m_LockHandle = AsInteger(lock);
    m_OwnsEndpoint = true;
    return true;
}

LocalIpcConnection
LocalIpcListener::Accept(std::string &error)
{
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
        error = SystemErrorMessage("CreateNamedPipe", GetLastError());
        return {};
    }

    if (!ConnectNamedPipe(pipe, nullptr)) {
        const DWORD lastError = GetLastError();
        if (lastError != ERROR_PIPE_CONNECTED) {
            error = SystemErrorMessage("ConnectNamedPipe", lastError);
            CloseHandle(pipe);
            return {};
        }
    }
    return LocalIpcConnection(AsInteger(pipe));
}

void
LocalIpcListener::Close()
{
    if (IsValidHandle(m_LockHandle)) {
        CloseHandle(AsHandle(m_LockHandle));
        m_LockHandle = -1;
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
    const std::string pipeName = NormalizeEndpoint(serviceName, endpoint);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMilliseconds);

    for (;;) {
        HANDLE pipe = CreateFileA(
            pipeName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) {
            return LocalIpcConnection(AsInteger(pipe));
        }

        const DWORD lastError = GetLastError();
        if (lastError != ERROR_PIPE_BUSY && lastError != ERROR_FILE_NOT_FOUND) {
            error = SystemErrorMessage("CreateFile(named pipe)", lastError);
            return {};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = SystemErrorMessage("timed out connecting to named pipe", lastError);
            return {};
        }
        if (lastError == ERROR_PIPE_BUSY) {
            WaitNamedPipeA(pipeName.c_str(), 100);
        } else {
            Sleep(25);
        }
    }
}

} // namespace Environment
