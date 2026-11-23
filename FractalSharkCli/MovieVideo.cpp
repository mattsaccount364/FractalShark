#include "stdafx.h"

#include "Exceptions.h"
#include "MovieJob.h"
#include "MovieVideo.h"

#include <algorithm>
#include <array>
#include <vector>

namespace FractalSharkCli {
namespace {
constexpr size_t PacketBytes = 188;
constexpr uint16_t PatPid = 0;
constexpr uint16_t PmtPid = 0x1000;
constexpr uint16_t VideoPid = 0x100;
constexpr uint64_t ClockHz = 90000;
constexpr uint64_t TimestampSpan = uint64_t{1} << 33;
constexpr uint64_t PresentationLead = ClockHz / 10;
constexpr uint64_t ClockInterval = ClockHz / 25;
constexpr uint64_t TableInterval = ClockHz / 10;
constexpr uint64_t CheckpointFrames = 32;
constexpr auto CheckpointInterval = std::chrono::seconds(1);

void
WriteClockBytes(unsigned char *to, uint64_t clock)
{
    // PCR base is in 90 kHz ticks. The extension is zero at this precision.
    to[0] = static_cast<unsigned char>(clock >> 25);
    to[1] = static_cast<unsigned char>(clock >> 17);
    to[2] = static_cast<unsigned char>(clock >> 9);
    to[3] = static_cast<unsigned char>(clock >> 1);
    to[4] = static_cast<unsigned char>((clock << 7) | 0x7e);
    to[5] = 0;
}

template <size_t N>
void
SealProgramTable(std::array<unsigned char, N> &table)
{
    // ISO/IEC 13818-1 PSI CRC-32 starts after the pointer field and operates
    // most-significant bit first. It protects tables, not video payloads.
    constexpr uint32_t polynomial = 0x04c11db7;
    uint32_t crc = UINT32_MAX;
    for (const auto byte : std::span(table).subspan(1, N - 5)) {
        crc ^= static_cast<uint32_t>(byte) << 24;
        for (unsigned bit = 0; bit < 8; ++bit) {
            const bool high = (crc & 0x80000000u) != 0;
            crc <<= 1;
            if (high) {
                crc ^= polynomial;
            }
        }
    }
    for (size_t byte = 0; byte < 4; ++byte) {
        table[N - 4 + byte] = static_cast<unsigned char>(crc >> (24 - 8 * byte));
    }
}
} // namespace

MovieVideo::MovieVideo(MovieJob &job)
    : m_Job(job), m_Partial(job.Directory() / "movie.ts.partial"),
      m_Target(job.Directory() / "movie.ts"), m_Next(job.NextFrame()), m_Bytes(job.VideoBytes())
{
    if (job.FrameCount() == 0 ||
        job.FrameCount() * ClockHz >= (TimestampSpan - PresentationLead) * job.Config().Video.Fps) {
        throw FractalSharkSeriousException("Movie exceeds the MPEG-TS timestamp span");
    }
    if (m_Bytes % PacketBytes != 0 || (m_Next == 0) != (m_Bytes == 0)) {
        throw FractalSharkSeriousException("Movie checkpoint does not describe a transport prefix");
    }
    const auto partialStatus = std::filesystem::symlink_status(m_Partial);
    const auto targetStatus = std::filesystem::symlink_status(m_Target);
    const bool hasPartial = std::filesystem::exists(partialStatus);
    if (hasPartial && !std::filesystem::is_regular_file(partialStatus)) {
        throw FractalSharkSeriousException("Movie partial output must be a regular file");
    }
    if (std::filesystem::exists(targetStatus)) {
        if (!std::filesystem::is_regular_file(targetStatus) || job.Remaining() != 0 ||
            std::filesystem::file_size(m_Target) != m_Bytes ||
            (hasPartial && !std::filesystem::equivalent(m_Partial, m_Target))) {
            throw FractalSharkSeriousException("Published movie does not match its checkpoint");
        }
        // Linux publishes a durable link before removing the partial name.
        if (hasPartial) {
            std::filesystem::remove(m_Partial);
        }
        m_Published = true;
        return;
    }
    const auto diskBytes = hasPartial ? std::filesystem::file_size(m_Partial) : 0;
    if (diskBytes < m_Bytes) {
        throw FractalSharkSeriousException("Movie partial output is shorter than its checkpoint");
    }
    if (diskBytes > m_Bytes) {
        std::filesystem::resize_file(m_Partial, m_Bytes);
        Environment::FlushMovieFile(m_Partial);
    }
}

size_t
MovieVideo::PidIndex(uint16_t pid) const
{
    return pid == PatPid ? 0 : pid == PmtPid ? 1 : 2;
}

void
MovieVideo::WritePacket(const std::array<unsigned char, PacketBytes> &packet)
{
    m_Output.write(reinterpret_cast<const char *>(packet.data()), packet.size());
    if (!m_Output) {
        throw FractalSharkSeriousException("Cannot write native movie output");
    }
    m_Bytes += packet.size();
}

void
MovieVideo::WritePackets(uint16_t pid,
                         std::span<const unsigned char> payload,
                         bool randomAccess,
                         std::optional<uint64_t> clock)
{
    const auto index = PidIndex(pid);
    bool start = true;
    while (!payload.empty()) {
        std::array<unsigned char, PacketBytes> packet;
        packet.fill(0xff);
        const bool discontinuity = m_FirstPacket[index];
        const unsigned flags =
            (discontinuity ? 0x80 : 0) | (randomAccess ? 0x40 : 0) | (clock ? 0x10 : 0);
        // A flagged adaptation field requires a length byte, a flags byte,
        // and the six PCR bytes when present. Short packets use stuffing.
        const size_t reserved = flags ? (clock ? 8 : 2) : 0;
        const size_t size = std::min(payload.size(), PacketBytes - 4 - reserved);
        const size_t adaptation = PacketBytes - 4 - size;
        packet[0] = 0x47;
        packet[1] = static_cast<unsigned char>((start ? 0x40 : 0) | (pid >> 8));
        packet[2] = static_cast<unsigned char>(pid);
        packet[3] = static_cast<unsigned char>((adaptation ? 0x30 : 0x10) | m_Continuity[index]);
        if (adaptation) {
            packet[4] = static_cast<unsigned char>(adaptation - 1);
            if (adaptation > 1) {
                packet[5] = static_cast<unsigned char>(flags);
            }
            if (clock) {
                WriteClockBytes(packet.data() + 6, *clock);
            }
        }
        std::copy_n(payload.begin(), size, packet.end() - size);
        WritePacket(packet);
        payload = payload.subspan(size);
        m_Continuity[index] = static_cast<uint8_t>((m_Continuity[index] + 1) & 15);
        m_FirstPacket[index] = false;
        start = randomAccess = false;
        clock.reset();
    }
}

void
MovieVideo::WriteClock(uint64_t clock)
{
    // Adaptation-only packets carry PCR during the gap between slow frames.
    // With no payload the continuity counter repeats the previous value.
    std::array<unsigned char, PacketBytes> packet;
    packet.fill(0xff);
    packet[0] = 0x47;
    packet[1] = static_cast<unsigned char>(VideoPid >> 8);
    packet[2] = static_cast<unsigned char>(VideoPid);
    packet[3] = static_cast<unsigned char>(0x20 | ((m_Continuity[PidIndex(VideoPid)] + 15) & 15));
    packet[4] = 183;
    packet[5] = 0x10;
    WriteClockBytes(packet.data() + 6, clock);
    WritePacket(packet);
    m_LastClock = clock;
}

void
MovieVideo::WriteProgram(uint64_t clock)
{
    // One program, one video stream, with PCR on that video PID.
    std::array<unsigned char, 17> pat{0,
                                      0x00,
                                      0xb0,
                                      13,
                                      0,
                                      1,
                                      0xc1,
                                      0,
                                      0,
                                      0,
                                      1,
                                      static_cast<unsigned char>(0xe0 | (PmtPid >> 8)),
                                      static_cast<unsigned char>(PmtPid)};
    const bool h264 = m_Job.Config().Video.Codec == FractalShark::MovieCodec::H264;
    std::array<unsigned char, 22> pmt{0,
                                      0x02,
                                      0xb0,
                                      18,
                                      0,
                                      1,
                                      0xc1,
                                      0,
                                      0,
                                      static_cast<unsigned char>(0xe0 | (VideoPid >> 8)),
                                      static_cast<unsigned char>(VideoPid),
                                      0xf0,
                                      0,
                                      static_cast<unsigned char>(h264 ? 0x1b : 0x24),
                                      static_cast<unsigned char>(0xe0 | (VideoPid >> 8)),
                                      static_cast<unsigned char>(VideoPid),
                                      0xf0,
                                      0};
    SealProgramTable(pat);
    SealProgramTable(pmt);
    WritePackets(PatPid, pat, false, std::nullopt);
    WritePackets(PmtPid, pmt, false, std::nullopt);
    m_LastProgram = clock;
}

void
MovieVideo::Append(uint64_t frame, std::span<const unsigned char> accessUnit, bool keyframe)
{
    if (m_Published || frame != m_Next || frame >= m_Job.FrameCount() || accessUnit.empty() ||
        (!m_LastClock && !keyframe)) {
        throw FractalSharkSeriousException("Invalid native movie access unit sequence");
    }
    if (!m_Output.is_open()) {
        m_Output.open(m_Partial, std::ios::binary | std::ios::app);
        if (!m_Output) {
            throw FractalSharkSeriousException("Cannot open native movie output");
        }
    }

    const auto clock = frame * ClockHz / m_Job.Config().Video.Fps;
    while (m_LastClock && *m_LastClock + ClockInterval < clock) {
        const auto nextClock = *m_LastClock + ClockInterval;
        if (nextClock - *m_LastProgram >= TableInterval) {
            WriteProgram(nextClock);
        }
        WriteClock(nextClock);
    }
    if (keyframe || !m_LastProgram || clock - *m_LastProgram >= TableInterval) {
        WriteProgram(clock);
    }

    // One Annex B access unit per video PES. Video PES in TS may use a zero
    // length when the access unit exceeds the 16-bit PES length field.
    const auto pts = clock + PresentationLead;
    const std::array<unsigned char, 14> header{0,
                                               0,
                                               1,
                                               0xe0,
                                               0,
                                               0,
                                               0x84,
                                               0x80,
                                               5,
                                               static_cast<unsigned char>(0x21 | ((pts >> 29) & 0x0e)),
                                               static_cast<unsigned char>(pts >> 22),
                                               static_cast<unsigned char>(1 | ((pts >> 14) & 0xfe)),
                                               static_cast<unsigned char>(pts >> 7),
                                               static_cast<unsigned char>(1 | ((pts << 1) & 0xfe))};
    std::vector<unsigned char> pes;
    pes.reserve(header.size() + accessUnit.size());
    pes.insert(pes.end(), header.begin(), header.end());
    pes.insert(pes.end(), accessUnit.begin(), accessUnit.end());
    const auto writePcr = !m_LastClock || clock - *m_LastClock >= ClockInterval;
    WritePackets(VideoPid, pes, keyframe, writePcr ? std::optional{clock} : std::nullopt);
    if (writePcr) {
        m_LastClock = clock;
    }
    ++m_Next;
    CheckpointIfDue();
}

void
MovieVideo::CheckpointIfDue()
{
    if (m_Next > m_Job.NextFrame() &&
        (m_Next - m_Job.NextFrame() >= CheckpointFrames ||
         std::chrono::steady_clock::now() - m_LastCommit >= CheckpointInterval)) {
        Checkpoint();
    }
}

void
MovieVideo::Checkpoint()
{
    if (m_Next == m_Job.NextFrame()) {
        return;
    }
    // Close before the platform opens its own flush handle. Publish the
    // checkpoint only after both stream bytes and their pathname are durable.
    m_Output.close();
    if (!m_Output) {
        throw FractalSharkSeriousException("Cannot close native movie output");
    }
    Environment::FlushMovieFile(m_Partial);
    m_Job.CommitVideo(m_Next, m_Bytes);
    m_LastCommit = std::chrono::steady_clock::now();
}

void
MovieVideo::Finish()
{
    Checkpoint();
    if (!m_Published && m_Job.Remaining() == 0) {
        Environment::PublishMovieFile(m_Partial, m_Target, false);
        m_Published = true;
    }
}
} // namespace FractalSharkCli
