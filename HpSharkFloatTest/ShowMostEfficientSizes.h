#pragma once
#include <cstdint>
#include <vector>

namespace SharkNTT {

// ================================================================
// Types
// ================================================================

struct Plateau {
    int n32_min;
    int n32_max;
    uint64_t bits_min;
    uint64_t bits_max;
    int b;
    int N;
    int L_min;
    int L_max;
};

// ================================================================
// Function prototypes
// ================================================================

// Build the plateau table for n32 = 1..maxN32.
std::vector<Plateau> BuildPrecisionPlateaus(int maxN32, int b_hint, int margin);

// Pretty-print the full plateau table.
void PrintPlateauTable(const std::vector<Plateau> &plateaus);

// Pretty-print the “tier” sweet spots (the right edges of the plateaus).
void PrintPrecisionTiers(const std::vector<Plateau> &plateaus);

} // namespace SharkNTT
#pragma once
