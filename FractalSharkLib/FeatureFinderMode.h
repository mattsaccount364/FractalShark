#pragma once

enum class FeatureFinderMode {
    Direct,
    PT,
    LA,

    DirectScan,
    PTScan,
    LAScan
};

// GPURef2 is the Ref2 CUDA implementation; append it to preserve the
// existing values of the original backends.
enum class NRInnerLoopBackend { GPU, CpuMT, CpuST, GPURef2 };

enum class NRCheckpointSavePolicy { Save, PreserveExisting };
