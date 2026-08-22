#pragma once

enum class FeatureFinderMode {
    Direct,
    PT,
    LA,

    DirectScan,
    PTScan,
    LAScan
};

enum class NRInnerLoopBackend { GPU, CpuMT, CpuST };

enum class NRCheckpointSavePolicy { Save, PreserveExisting };
