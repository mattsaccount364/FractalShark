#pragma once

#include <cstdint>
#include <string>

namespace HpShark {

struct LaunchParams {
    LaunchParams(int32_t numBlocksIn, int32_t threadsPerBlockIn)
        : NumBlocks{numBlocksIn}, ThreadsPerBlock{threadsPerBlockIn},
          TotalThreads{numBlocksIn * threadsPerBlockIn}
    {
    }

    LaunchParams()
        : NumBlocks{rand() % 128 + 1}, ThreadsPerBlock{32 * (rand() % 8 + 1)},
          TotalThreads{NumBlocks * ThreadsPerBlock}
    {
    }

    int32_t NumBlocks;
    int32_t ThreadsPerBlock;
    int32_t TotalThreads;

    std::string
    ToString() const
    {
        return std::string("Blocks: ") + std::to_string(NumBlocks) +
               ", ThreadsPerBlock: " + std::to_string(ThreadsPerBlock) +
               ", TotalThreads: " + std::to_string(TotalThreads);
    }
};

} // namespace HpShark