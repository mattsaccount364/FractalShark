#include "HpSharkFloat.h"
#include "MultiplyNTTPlanBuilder.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// Your existing PlanPrime + BuildPlanPrime
// (use your real definitions; trimmed here just for context)
// -----------------------------------------------------------------------------
namespace SharkNTT {

// -----------------------------------------------------------------------------
// Tier / plateau finder
// -----------------------------------------------------------------------------

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

// Build the plateau table up to maxN32
std::vector<Plateau>
BuildPrecisionPlateaus(int maxN32, int b_hint, int margin)
{
    std::vector<Plateau> plateaus;

    bool havePlateau = false;
    int plateauStartN32 = 0;
    PlanPrime plateauPlan{}; // plan for this plateau (b, N, etc.)

    for (int n32 = 1; n32 <= maxN32; ++n32) {
        PlanPrime plan = BuildPlanPrime(n32, b_hint, margin);

        if (!plan.ok) {
            // If we were in a plateau and we hit a bad plan, close it
            if (havePlateau) {
                Plateau p{};
                p.n32_min = plateauStartN32;
                p.n32_max = n32 - 1;
                p.bits_min = static_cast<uint64_t>(p.n32_min) * 32ull;
                p.bits_max = static_cast<uint64_t>(p.n32_max) * 32ull;
                p.b = plateauPlan.b;
                p.N = plateauPlan.N;

                // L at the endpoints
                uint64_t totalBitsMin = static_cast<uint64_t>(p.n32_min) * 32ull;
                uint64_t totalBitsMax = static_cast<uint64_t>(p.n32_max) * 32ull;
                p.L_min = static_cast<int>(
                    CeilDivU32(static_cast<uint32_t>(totalBitsMin), static_cast<uint32_t>(p.b)));
                p.L_max = static_cast<int>(
                    CeilDivU32(static_cast<uint32_t>(totalBitsMax), static_cast<uint32_t>(p.b)));

                plateaus.push_back(p);
                havePlateau = false;
            }
            continue;
        }

        // plan.ok == true
        if (!havePlateau) {
            // Start a new plateau
            havePlateau = true;
            plateauStartN32 = n32;
            plateauPlan = plan;
        } else {
            // We are already in a plateau: check if (b, N) changed
            if (plan.b != plateauPlan.b || plan.N != plateauPlan.N) {
                // Close previous plateau at n32 - 1
                Plateau p{};
                p.n32_min = plateauStartN32;
                p.n32_max = n32 - 1;
                p.bits_min = static_cast<uint64_t>(p.n32_min) * 32ull;
                p.bits_max = static_cast<uint64_t>(p.n32_max) * 32ull;
                p.b = plateauPlan.b;
                p.N = plateauPlan.N;

                uint64_t totalBitsMin = static_cast<uint64_t>(p.n32_min) * 32ull;
                uint64_t totalBitsMax = static_cast<uint64_t>(p.n32_max) * 32ull;
                p.L_min = static_cast<int>(
                    CeilDivU32(static_cast<uint32_t>(totalBitsMin), static_cast<uint32_t>(p.b)));
                p.L_max = static_cast<int>(
                    CeilDivU32(static_cast<uint32_t>(totalBitsMax), static_cast<uint32_t>(p.b)));

                plateaus.push_back(p);

                // Start new plateau
                plateauStartN32 = n32;
                plateauPlan = plan;
            } else {
                // Same plateau; just extend it
                plateauPlan = plan; // keep last plan (for N, L, etc.)
            }
        }
    }

    // Close trailing plateau if it’s still open
    if (havePlateau) {
        Plateau p{};
        p.n32_min = plateauStartN32;
        p.n32_max = maxN32;
        p.bits_min = static_cast<uint64_t>(p.n32_min) * 32ull;
        p.bits_max = static_cast<uint64_t>(p.n32_max) * 32ull;
        p.b = plateauPlan.b;
        p.N = plateauPlan.N;

        uint64_t totalBitsMin = static_cast<uint64_t>(p.n32_min) * 32ull;
        uint64_t totalBitsMax = static_cast<uint64_t>(p.n32_max) * 32ull;
        p.L_min = static_cast<int>(
            CeilDivU32(static_cast<uint32_t>(totalBitsMin), static_cast<uint32_t>(p.b)));
        p.L_max = static_cast<int>(
            CeilDivU32(static_cast<uint32_t>(totalBitsMax), static_cast<uint32_t>(p.b)));

        plateaus.push_back(p);
    }

    return plateaus;
}

// Convenience: compute N/(2L) as double
static double
overhead(double N, double L)
{
    return N / (2.0 * L);
}

// -----------------------------------------------------------------------------
// Dump functions
// -----------------------------------------------------------------------------

void
PrintPlateauTable(const std::vector<Plateau> &plateaus)
{
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Idx  n32_min–n32_max   bits_min–bits_max     b    N     "
                 "L_min-L_max   overhead_start-->end\n";

    int idx = 0;
    for (const auto &p : plateaus) {
        double ov_start = overhead(p.N, p.L_min);
        double ov_end = overhead(p.N, p.L_max);

        std::cout << std::setw(3) << idx++ << "  " << std::setw(6) << p.n32_min << "–" << std::setw(6)
                  << p.n32_max << "   " << std::setw(8) << p.bits_min << "–" << std::setw(8)
                  << p.bits_max << "   " << std::setw(2) << p.b << "  " << std::setw(6) << p.N << "   "
                  << std::setw(6) << p.L_min << "–" << std::setw(6) << p.L_max << "   " << ov_start
                  << "-->" << ov_end << "\n";
    }
}

// Print just the "tier" sweet spots (right edge of each plateau)
void
PrintPrecisionTiers(const std::vector<Plateau> &plateaus)
{
    std::cout << "Tier  n32   bits   b   N   L_max   overhead\n";
    int tier = 1;
    for (const auto &p : plateaus) {
        int n32 = p.n32_max;
        uint64_t bits = p.bits_max;
        double ov = overhead(p.N, p.L_max);

        std::cout << std::setw(4) << tier++ << "  " << std::setw(6) << n32 << "  " << std::setw(8)
                  << bits << "  " << std::setw(2) << p.b << "  " << std::setw(6) << p.N << "  "
                  << std::setw(6) << p.L_max << "  " << ov << "\n";
    }
}

} // namespace SharkNTT
