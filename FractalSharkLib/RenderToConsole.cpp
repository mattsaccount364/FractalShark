#include "stdafx.h"

#include "RenderToConsole.h"

#include "Fractal.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace {

// Characters sorted roughly by ascending visual density (ink coverage).
// Normalized mapping spreads the full range of exterior iteration values
// across this ramp for maximum contrast.  Set interior is handled separately.
constexpr char DensityRamp[] = ".:-=+*#%@";
constexpr size_t RampLength = sizeof(DensityRamp) - 1; // exclude NUL

// Sentinel: this block is set-interior (all pixels == maxIter).
constexpr double SetInteriorSentinel = -1.0;

// ANSI 256-color: map a normalized 0..1 value to one of 216 color-cube
// entries (codes 16–231).  We cycle through hues at full saturation/value.
int
AnsiColorCode(double normalized)
{
    // 216 color cube: 6×6×6 (R,G,B each 0–5), codes 16..231.
    int slot = static_cast<int>(normalized * 215.0);
    slot = std::clamp(slot, 0, 215);

    // Walk the cube diagonally through hues.
    int sector = slot / 36;
    int pos = slot % 36;
    int v = pos * 5 / 35; // 0..5

    int r, g, b;
    switch (sector) {
        case 0:
            r = 5;
            g = v;
            b = 0;
            break; // red → yellow
        case 1:
            r = 5 - v;
            g = 5;
            b = 0;
            break; // yellow → green
        case 2:
            r = 0;
            g = 5;
            b = v;
            break; // green → cyan
        case 3:
            r = 0;
            g = 5 - v;
            b = 5;
            break; // cyan → blue
        case 4:
            r = v;
            g = 0;
            b = 5;
            break; // blue → magenta
        default:
            r = 5;
            g = 0;
            b = 5 - v;
            break; // magenta → red
    }

    return 16 + 36 * r + 6 * g + b;
}

} // namespace

void
RenderToConsole(const Fractal &fractal, const ConsoleRenderOptions &opts, std::ostream &out)
{
    const auto &iters = fractal.GetCurIters();
    const size_t srcW = iters.m_OutputWidth;
    const size_t srcH = iters.m_OutputHeight;

    if (srcW == 0 || srcH == 0) {
        return;
    }

    const size_t consW = std::max<size_t>(opts.ConsoleWidth, 1);
    const size_t consH = std::max<size_t>(opts.ConsoleHeight, 1);

    // Aspect-ratio correction: terminal characters are roughly 2:1 (height:width).
    // Each console row represents twice as many source pixels vertically as
    // each column does horizontally, so the output looks proportional.
    const double cellW = static_cast<double>(srcW) / static_cast<double>(consW);
    const double cellH = static_cast<double>(srcH) / static_cast<double>(consH);

    const IterTypeFull maxIter = fractal.GetNumIterationsRT();

    // --- Pass 1: compute block averages and global min/max ---
    std::vector<double> blockAvg(consW * consH, SetInteriorSentinel);
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = 0.0;

    for (size_t row = 0; row < consH; row++) {
        const size_t srcY0 = static_cast<size_t>(row * cellH);
        const size_t srcY1 = std::min(static_cast<size_t>((row + 1) * cellH), srcH);

        for (size_t col = 0; col < consW; col++) {
            const size_t srcX0 = static_cast<size_t>(col * cellW);
            const size_t srcX1 = std::min(static_cast<size_t>((col + 1) * cellW), srcW);

            double sum = 0.0;
            size_t count = 0;
            for (size_t y = srcY0; y < srcY1; y++) {
                for (size_t x = srcX0; x < srcX1; x++) {
                    IterTypeFull val = iters.GetItersArrayValSlow(x, y);
                    if (val < maxIter) {
                        sum += static_cast<double>(val);
                        count++;
                    }
                }
            }

            if (count > 0) {
                double avg = sum / static_cast<double>(count);
                avg = std::max(avg, 1.0); // clamp to 1 for log safety
                blockAvg[row * consW + col] = avg;
                globalMin = std::min(globalMin, avg);
                globalMax = std::max(globalMax, avg);
            }
            // else: stays SetInteriorSentinel (all pixels were maxIter or empty)
        }
    }

    // Check for degenerate case: no exterior pixels at all.
    if (globalMax <= 0.0) {
        out << "(All pixels are set-interior — nothing to display)\n";
        out.flush();
        return;
    }

    // Check for uniform iteration values (e.g., precision mismatch).
    if (globalMin >= globalMax) {
        out << "(Warning: all exterior pixels have the same iteration count "
            << static_cast<size_t>(globalMin) << " — try a lower zoom or different algorithm)\n";
    }

    // Decide whether to use log-scale: if the range spans at least 2×,
    // log-scale gives better contrast.  Otherwise use linear.
    const bool useLog = (globalMax > globalMin * 2.0);
    const double logMin = useLog ? std::log(globalMin) : globalMin;
    const double logMax = useLog ? std::log(globalMax) : globalMax;
    const double logRange = logMax - logMin;

    // --- Pass 2: normalize and emit characters ---
    std::string line;
    line.reserve(consW + 16);

    for (size_t row = 0; row < consH; row++) {
        line.clear();

        for (size_t col = 0; col < consW; col++) {
            double avg = blockAvg[row * consW + col];

            // Set interior or empty block.
            if (avg == SetInteriorSentinel) {
                if (opts.Color) {
                    line += "\033[48;5;0m \033[0m";
                } else {
                    line += ' ';
                }
                continue;
            }

            // Normalize to 0..1.
            double normalized = 0.0;
            if (logRange > 0.0) {
                double logVal = useLog ? std::log(avg) : avg;
                normalized = (logVal - logMin) / logRange;
                normalized = std::clamp(normalized, 0.0, 1.0);
            }

            size_t rampIdx = static_cast<size_t>(normalized * static_cast<double>(RampLength - 1) + 0.5);
            rampIdx = std::min(rampIdx, RampLength - 1);
            char ch = DensityRamp[rampIdx];

            if (opts.Color) {
                int code = AnsiColorCode(normalized);
                line += "\033[38;5;";
                line += std::to_string(code);
                line += 'm';
                line += ch;
                line += "\033[0m";
            } else {
                line += ch;
            }
        }

        out << line << '\n';
    }

    out.flush();
}
