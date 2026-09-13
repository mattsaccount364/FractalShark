#pragma once

namespace HpShark {

struct TestParams {
    enum class MpirThreadingMode {
        Disabled,
        MultiThreaded,
        SingleThreaded,
    };

    enum class VerboseMode {
        None = 0,
        Debug = 1,
    };

    MpirThreadingMode m_MpirThreading = MpirThreadingMode::Disabled;
    bool m_TestReferenceImpl = false;
    bool m_TestInfiniteCorrectness = true;
    VerboseMode m_VerboseMode = VerboseMode::None;

    bool
    IsMpirEnabled() const
    {
        return m_MpirThreading != MpirThreadingMode::Disabled;
    }

    bool
    UseMpirMultithreading() const
    {
        return m_MpirThreading == MpirThreadingMode::MultiThreaded;
    }

    bool
    IsVerbose() const
    {
        return m_VerboseMode == VerboseMode::Debug;
    }
};

} // namespace HpShark
