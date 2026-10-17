#include "GpuPrecisionDispatch.h"
#include "HighPrecision.h"
#include "KernelInvoke.h"
#include "RenderTestSupport.h"
#include "TestFramework.h"

#include <fstream>
#include <vector>

namespace {
template <class Family>
void
CheckReferenceBucket(uint32_t limbs, const std::string &caseId)
{
    RenderTests::ScopedDirectory directory{caseId};
    DispatchByLimbCount<Family>(limbs, [&]<class Params> {
        const HighPrecision cx{"0.125"};
        const HighPrecision cy{"0.25"};
        HpShark::GpuOrbitSession<Params> session{HpShark::LaunchParams{0, 0},
                                                 typename Params::Float{1e-30},
                                                 cx.backend(),
                                                 cy.backend(),
                                                 GetReferenceEffectivePrecisionLimbs(limbs, limbs),
                                                 nullptr};
        session.InvokeChunk(8);
        const auto &results = session.GetResults();
        ASSERT_EQ(results.OutputIterCount, uint64_t{8});
        ASSERT_TRUE(results.PeriodicityStatus == PeriodicityResult::Continue);
        std::string canonical;
        for (uint64_t i = 0; i < results.OutputIterCount; ++i) {
            canonical += HdrToString<true>(results.OutputIters[i].x) + "\n";
            canonical += HdrToString<true>(results.OutputIters[i].y) + "\n";
        }
        std::ofstream output("reference.txt");
        output << canonical;
        ASSERT_TRUE(output.good());
        const std::vector<uint8_t> bytes(canonical.begin(), canonical.end());
        RenderTests::CheckChecksum(caseId, bytes);
    });
}

template <class Family>
void
CheckReferencePeriodicityNormalization(bool splitChunks)
{
    DispatchByLimbCount<Family>(MinSupportedLimbCount, [&]<class Params> {
        // All values are exact dyadics in every output family. At z1=c, the strict
        // comparison is equality and must continue. At z2=33/64, d2=7/4 gives
        // threshold 21/32. Its product mantissa needs normalization before comparison.
        const HighPrecision cx{"0.375"};
        const HighPrecision cy{"0"};
        HpShark::GpuOrbitSession<Params> session{
            HpShark::LaunchParams{0, 0},
            typename Params::Float{0.1875},
            cx.backend(),
            cy.backend(),
            GetReferenceEffectivePrecisionLimbs(MinSupportedLimbCount, MinSupportedLimbCount),
            nullptr};
        if (splitChunks) {
            session.InvokeChunk(1);
            const auto &first = session.GetResults();
            ASSERT_EQ(first.OutputIterCount, uint64_t{1});
            ASSERT_TRUE(first.PeriodicityStatus == PeriodicityResult::Continue);
            ASSERT_TRUE(first.OutputIters[0].x == typename Params::Float{0.375});
            ASSERT_TRUE(first.OutputIters[0].y == typename Params::Float{0});
        }
        session.InvokeChunk(8);
        const auto &results = session.GetResults();
        ASSERT_TRUE(results.PeriodicityStatus == PeriodicityResult::PeriodFound);
        ASSERT_EQ(results.OutputIterCount, splitChunks ? uint64_t{1} : uint64_t{2});
        if (!splitChunks) {
            ASSERT_TRUE(results.OutputIters[0].x == typename Params::Float{0.375});
            ASSERT_TRUE(results.OutputIters[0].y == typename Params::Float{0});
        }
        const auto last = results.OutputIterCount - 1;
        ASSERT_TRUE(results.OutputIters[last].x == typename Params::Float{0.515625});
        ASSERT_TRUE(results.OutputIters[last].y == typename Params::Float{0});
        ASSERT_TRUE(results.DzdcX == typename Params::Float{1.75});
        // Zero arithmetic may retain an exponent different from a default HDR zero.
        // Verify its numeric value rather than requiring identical representations.
        ASSERT_TRUE(results.DzdcY.getMantissa() == typename Params::SubType{0.0f});
    });
}

template <class Family>
void
CheckReferencePeriodicityStrictBoundary()
{
    DispatchByLimbCount<Family>(MinSupportedLimbCount, [&]<class Params> {
        // For c=1/2, z2=3/4 and d2=2. These radii place 2*R*d2 below,
        // exactly on, and above z2 without triggering the first iteration's check.
        for (const double radius : {0.15625, 0.1875, 0.21875}) {
            const HighPrecision cx{"0.5"};
            const HighPrecision cy{"0"};
            HpShark::GpuOrbitSession<Params> session{
                HpShark::LaunchParams{0, 0},
                typename Params::Float{radius},
                cx.backend(),
                cy.backend(),
                GetReferenceEffectivePrecisionLimbs(MinSupportedLimbCount, MinSupportedLimbCount),
                nullptr};
            session.InvokeChunk(2);
            const auto &results = session.GetResults();
            ASSERT_EQ(results.OutputIterCount, uint64_t{2});
            ASSERT_TRUE(results.OutputIters[0].x == typename Params::Float{0.5});
            ASSERT_TRUE(results.OutputIters[1].x == typename Params::Float{0.75});
            const auto expected =
                radius > 0.1875 ? PeriodicityResult::PeriodFound : PeriodicityResult::Continue;
            ASSERT_TRUE(results.PeriodicityStatus == expected);
        }
    });
}

template <class Family>
void
RegisterReferencePeriodicityTests(const std::string &family)
{
    for (const bool splitChunks : {false, true}) {
        TestFramework::RegisterCase(
            "GpuReferencePeriodicity_NormalizedThreshold_" + family +
                (splitChunks ? "_SplitChunks" : "_SingleChunk"),
            [splitChunks] { CheckReferencePeriodicityNormalization<Family>(splitChunks); },
            true,
            "",
            false);
    }
    TestFramework::RegisterCase(
        "GpuReferencePeriodicity_StrictBoundary_" + family,
        [] { CheckReferencePeriodicityStrictBoundary<Family>(); },
        true,
        "",
        false);
}

const bool registered = [] {
    RegisterReferencePeriodicityTests<SharkParamsBaseFamily>("Float");
    RegisterReferencePeriodicityTests<SharkParamsDblFamily>("Double");
    RegisterReferencePeriodicityTests<SharkParamsDbfFamily>("Dblflt");
    for (uint32_t limbs = MinSupportedLimbCount; limbs <= MaxSupportedLimbCount; limbs *= 2) {
        for (const int family : {0, 1, 2}) {
            const auto name = "RenderGolden_GpuReferenceBucket_" + std::to_string(limbs) +
                              (family == 0   ? "_Float"
                               : family == 1 ? "_Double"
                                             : "_Dblflt");
            TestFramework::RegisterCase(
                name,
                [=] {
                    switch (family) {
                        case 0:
                            CheckReferenceBucket<SharkParamsBaseFamily>(limbs, name);
                            break;
                        case 1:
                            CheckReferenceBucket<SharkParamsDblFamily>(limbs, name);
                            break;
                        case 2:
                            CheckReferenceBucket<SharkParamsDbfFamily>(limbs, name);
                            break;
                    }
                },
                true,
                "",
                true);
        }
    }
    return true;
}();
} // namespace
