#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace FractalSharkCli {

// Streaming SHA-256 (FIPS 180-4). A chunk digest covers the concatenated RGB8
// frames, so verification needs constant memory and only one digest per chunk.
class MovieHash {
public:
    void
    Update(std::span<const unsigned char> bytes)
    {
        m_Bytes += bytes.size();
        for (const auto byte : bytes) {
            m_Block[m_Used++] = byte;
            if (m_Used == m_Block.size()) {
                Transform();
                m_Used = 0;
            }
        }
    }
    void
    Update(std::string_view text)
    {
        Update(std::span<const unsigned char>(reinterpret_cast<const unsigned char *>(text.data()),
                                              text.size()));
    }
    std::string
    Finish() const
    {
        auto result = *this;
        const uint64_t bitCount = m_Bytes * 8;
        const unsigned char marker = 0x80;
        result.Update(std::span<const unsigned char>(&marker, 1));
        const unsigned char zero = 0;
        while (result.m_Used != 56) {
            result.Update(std::span<const unsigned char>(&zero, 1));
        }
        std::array<unsigned char, 8> length;
        for (size_t i = 0; i < length.size(); ++i) {
            length[i] = static_cast<unsigned char>(bitCount >> (56 - 8 * i));
        }
        result.Update(length);
        constexpr std::string_view Hex = "0123456789abcdef";
        std::string digest;
        for (const auto word : result.m_State) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                digest += Hex[(word >> shift) & 15];
            }
        }
        return digest;
    }
    static std::string
    Digest(std::string_view text)
    {
        MovieHash hash;
        hash.Update(text);
        return hash.Finish();
    }

private:
    void
    Transform()
    {
        static constexpr std::array<uint32_t, 64> Constants{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::array<uint32_t, 64> words;
        for (size_t i = 0; i < 16; ++i) {
            words[i] = (uint32_t{m_Block[4 * i]} << 24) | (uint32_t{m_Block[4 * i + 1]} << 16) |
                       (uint32_t{m_Block[4 * i + 2]} << 8) | m_Block[4 * i + 3];
        }
        for (size_t i = 16; i < words.size(); ++i) {
            const auto s0 =
                std::rotr(words[i - 15], 7) ^ std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
            const auto s1 =
                std::rotr(words[i - 2], 17) ^ std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        auto state = m_State;
        for (size_t i = 0; i < words.size(); ++i) {
            const auto s1 = std::rotr(state[4], 6) ^ std::rotr(state[4], 11) ^ std::rotr(state[4], 25);
            const auto choice = (state[4] & state[5]) ^ (~state[4] & state[6]);
            const auto t1 = state[7] + s1 + choice + Constants[i] + words[i];
            const auto s0 = std::rotr(state[0], 2) ^ std::rotr(state[0], 13) ^ std::rotr(state[0], 22);
            const auto majority = (state[0] & state[1]) ^ (state[0] & state[2]) ^ (state[1] & state[2]);
            for (size_t j = 7; j > 0; --j) {
                state[j] = state[j - 1];
            }
            state[4] += t1;
            state[0] = t1 + s0 + majority;
        }
        for (size_t i = 0; i < state.size(); ++i) {
            m_State[i] += state[i];
        }
    }
    std::array<uint32_t, 8> m_State{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<unsigned char, 64> m_Block{};
    size_t m_Used = 0;
    uint64_t m_Bytes = 0;
};
} // namespace FractalSharkCli
