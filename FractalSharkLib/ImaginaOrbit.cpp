#include "stdafx.h"
#include "ImaginaOrbit.h"

namespace Imagina {

    ImaginaATInfo::ImaginaATInfo() :
        StepLength{},
        ThresholdC{},
        SqrEscapeRadius{},
        RefC{},
        ZCoeff{},
        CCoeff{},
        InvZCoeff{} {
    }

    LAReferenceTrivialContent::LAReferenceTrivialContent() :
        Refc{},
        RefIt{},
        MaxIt{},
        DoublePrecisionPT{},
        DirectEvaluate{},
        IsPeriodic{},
        UseAT{},
        AT{},
        LAStageCount{} {
    }

    LAReferenceTrivialContent::LAReferenceTrivialContent(
        const std::complex<SRReal> &refc,
        size_t refIt,
        size_t maxIt,
        bool doublePrecisionPT,
        bool directEvaluate,
        bool isPeriodic,
        bool useAT,
        const ImaginaATInfo &at,
        size_t laStageCount) :
        Refc{ refc },
        RefIt{ refIt },
        MaxIt{ maxIt },
        DoublePrecisionPT{ doublePrecisionPT },
        DirectEvaluate{ directEvaluate },
        IsPeriodic{ isPeriodic },
        UseAT{ useAT },
        AT{ at },
        LAStageCount{ laStageCount } {
    }

} // namespace Imagina