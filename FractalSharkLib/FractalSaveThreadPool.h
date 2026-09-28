#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class FractalSaveThreadPool {
public:
    using Task = std::move_only_function<void()>;
    using TaskFactory = std::function<Task()>;
    using MemoryLoadFunction = std::function<uint32_t()>;

    FractalSaveThreadPool(size_t maxWorkers, MemoryLoadFunction getMemoryLoad);
    ~FractalSaveThreadPool();

    FractalSaveThreadPool(const FractalSaveThreadPool &) = delete;
    FractalSaveThreadPool &operator=(const FractalSaveThreadPool &) = delete;

    void Submit(const TaskFactory &makeTask);
    bool Cleanup(bool all);
    void Shutdown();

private:
    void WorkerLoop();

    const size_t m_MaxWorkers;
    const MemoryLoadFunction m_GetMemoryLoad;
    std::mutex m_Mutex;
    std::condition_variable m_WorkAvailable;
    std::condition_variable m_SlotAvailable;
    std::deque<Task> m_Queue;
    std::vector<std::thread> m_Workers;
    size_t m_Outstanding = 0;
    size_t m_UnreportedCompletions = 0;
    bool m_Accepting = true;
    bool m_Stopping = false;
};
