#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace FractalSharkCli {

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

class LocalConnection {
public:
    LocalConnection() = default;
    ~LocalConnection();

    LocalConnection(const LocalConnection &) = delete;
    LocalConnection &operator=(const LocalConnection &) = delete;

    LocalConnection(LocalConnection &&other) noexcept;
    LocalConnection &operator=(LocalConnection &&other) noexcept;

    // The native handle is intentionally represented as an integer so this
    // header stays independent of Windows and POSIX socket types.
    explicit LocalConnection(std::intptr_t nativeHandle);

    bool IsOpen() const;
    bool ReadExact(void *buffer, size_t size, std::string &error);
    bool WriteExact(const void *buffer, size_t size, std::string &error);
    void Close();

private:
    std::intptr_t m_NativeHandle = -1;
};

class LocalListener {
public:
    LocalListener() = default;
    ~LocalListener();

    LocalListener(const LocalListener &) = delete;
    LocalListener &operator=(const LocalListener &) = delete;
    LocalListener(LocalListener &&) = delete;
    LocalListener &operator=(LocalListener &&) = delete;

    bool Open(std::string_view endpoint, std::string &error);
    LocalConnection Accept(std::string &error);
    void Close();

    const std::string &Endpoint() const;

private:
    std::intptr_t m_NativeHandle = -1;
    std::intptr_t m_LockHandle = -1;
    bool m_OwnsEndpoint = false;
    std::string m_Endpoint;
};

std::string DefaultEndpoint();
std::string NormalizeEndpoint(std::string_view endpoint);

bool SendRequest(std::string_view endpoint,
                 const IpcRequest &request,
                 IpcResponse &response,
                 std::string &error);
bool ReadRequest(LocalConnection &connection, IpcRequest &request, std::string &error);
bool WriteResponse(LocalConnection &connection, const IpcResponse &response, std::string &error);

} // namespace FractalSharkCli
