#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>

namespace FractalSharkCli {
class MovieJob;

// Writes one video program in 188-byte MPEG-TS packets. Each append ends
// at an access-unit boundary suitable for a durable resume checkpoint.
class MovieVideo {
public:
    explicit MovieVideo(MovieJob &job);
    void Append(uint64_t frame, std::span<const unsigned char> accessUnit, bool keyframe);
    void CheckpointIfDue();
    void Finish();

private:
    void WriteProgram(uint64_t clock);
    void WriteClock(uint64_t clock);
    void WritePackets(uint16_t pid,
                      std::span<const unsigned char> payload,
                      bool randomAccess,
                      std::optional<uint64_t> clock);
    void WritePacket(const std::array<unsigned char, 188> &packet);
    void Checkpoint();
    size_t PidIndex(uint16_t pid) const;

    MovieJob &m_Job;
    std::filesystem::path m_Partial;
    std::filesystem::path m_Target;
    std::ofstream m_Output;
    uint64_t m_Next;
    uint64_t m_Bytes;
    std::array<uint8_t, 3> m_Continuity{};
    std::array<bool, 3> m_FirstPacket{true, true, true};
    std::optional<uint64_t> m_LastProgram;
    std::optional<uint64_t> m_LastClock;
    std::chrono::steady_clock::time_point m_LastCommit = std::chrono::steady_clock::now();
    bool m_Published = false;
};
} // namespace FractalSharkCli
