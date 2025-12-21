#pragma once

namespace FractalShark {

// RAII wrapper for HMENU.
class UniqueHMenu final {
public:
    UniqueHMenu() = default;
    explicit UniqueHMenu(HMENU h) noexcept : h_(h) {}

    UniqueHMenu(const UniqueHMenu &) = delete;
    UniqueHMenu &operator=(const UniqueHMenu &) = delete;

    UniqueHMenu(UniqueHMenu &&other) noexcept : h_(std::exchange(other.h_, nullptr)) {}
    UniqueHMenu &operator=(UniqueHMenu &&other) noexcept;

    ~UniqueHMenu();

    void reset(HMENU h = nullptr) noexcept;
    HMENU
    get() const noexcept { return h_; }
    explicit
    operator bool() const noexcept
    {
        return h_ != nullptr;
    }

private:
    HMENU h_ = nullptr;
};

} // namespace FractalShark
