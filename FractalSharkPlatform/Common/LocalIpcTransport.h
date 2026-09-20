#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "LocalIpcTransportDetail.h"

namespace Environment {

class LocalIpcConnection {
public:
    LocalIpcConnection() = default;
    explicit LocalIpcConnection(std::intptr_t nativeHandle);
    ~LocalIpcConnection();

    LocalIpcConnection(const LocalIpcConnection &) = delete;
    LocalIpcConnection &operator=(const LocalIpcConnection &) = delete;

    LocalIpcConnection(LocalIpcConnection &&other) noexcept;
    LocalIpcConnection &operator=(LocalIpcConnection &&other) noexcept;

    bool IsOpen() const noexcept;
    bool ReadExact(void *buffer, size_t size, std::string &error);
    bool WriteExact(const void *buffer, size_t size, std::string &error);
    void Close() noexcept;

private:
    std::intptr_t m_NativeHandle = LocalIpcDetail::InvalidNativeHandle;
};

class LocalIpcListener {
public:
    LocalIpcListener() = default;
    ~LocalIpcListener();

    LocalIpcListener(const LocalIpcListener &) = delete;
    LocalIpcListener &operator=(const LocalIpcListener &) = delete;
    LocalIpcListener(LocalIpcListener &&) = delete;
    LocalIpcListener &operator=(LocalIpcListener &&) = delete;

    bool Open(std::string_view serviceName, std::string_view endpoint, std::string &error);
    LocalIpcConnection Accept(std::string &error);
    void Close() noexcept;

    std::string_view Endpoint() const noexcept;

private:
    std::intptr_t m_NativeHandle = LocalIpcDetail::InvalidNativeHandle;
    std::intptr_t m_LockHandle = LocalIpcDetail::InvalidNativeHandle;
    std::string m_Endpoint;
};

LocalIpcConnection ConnectLocalIpc(std::string_view serviceName,
                                   std::string_view endpoint,
                                   uint32_t timeoutMilliseconds,
                                   std::string &error);

} // namespace Environment
