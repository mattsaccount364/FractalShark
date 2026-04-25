// CRC-64 (ECMA-182, polynomial 0x42F0E1EBA9EA3693). Hand-rolled,
// header-only, no external dependencies. Used by golden-render tests
// in TestRenderGoldens.cpp.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

class Crc64 {
public:
    static uint64_t Compute(const void *data, size_t len) {
        const uint64_t *table = Table();
        uint64_t crc = 0;
        const auto *p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < len; ++i) {
            crc = table[static_cast<uint8_t>(crc >> 56) ^ p[i]] ^ (crc << 8);
        }
        return crc;
    }

    static std::string ToHex(uint64_t value) {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out(16, '0');
        for (int i = 15; i >= 0; --i) {
            out[i] = kHex[value & 0xF];
            value >>= 4;
        }
        return out;
    }

private:
    static const uint64_t *Table() {
        static const auto table = []() {
            std::array<uint64_t, 256> t{};
            constexpr uint64_t kPoly = 0x42F0E1EBA9EA3693ULL;
            for (uint32_t i = 0; i < 256; ++i) {
                uint64_t c = static_cast<uint64_t>(i) << 56;
                for (int k = 0; k < 8; ++k) {
                    if (c & (1ULL << 63)) {
                        c = (c << 1) ^ kPoly;
                    } else {
                        c <<= 1;
                    }
                }
                t[i] = c;
            }
            return t;
        }();
        return table.data();
    }
};
