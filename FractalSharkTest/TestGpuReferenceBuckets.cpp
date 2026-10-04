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

const bool registered = [] {
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
