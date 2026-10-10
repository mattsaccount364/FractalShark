#include "stdafx.h"

#include "FractalSaveThreadPool.h"

#include "ConsoleLog.h"
#include "Environment.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace {

class OutstandingReservation {
public:
    OutstandingReservation(std::mutex &mutex,
                           std::condition_variable &slotAvailable,
                           size_t &outstanding) noexcept
        : m_Mutex(mutex), m_SlotAvailable(slotAvailable), m_Outstanding(outstanding)
    {
    }

    OutstandingReservation(const OutstandingReservation &) = delete;
    OutstandingReservation &operator=(const OutstandingReservation &) = delete;

    ~OutstandingReservation()
    {
        if (!m_Queued) {
            {
                std::lock_guard lock(m_Mutex);
                --m_Outstanding;
            }
            m_SlotAvailable.notify_all();
        }
    }

    void
    Commit() noexcept
    {
        m_Queued = true;
    }

private:
    std::mutex &m_Mutex;
    std::condition_variable &m_SlotAvailable;
    size_t &m_Outstanding;
    bool m_Queued = false;
};

} // namespace

FractalSaveThreadPool::FractalSaveThreadPool(size_t maxWorkers, MemoryLoadFunction getMemoryLoad)
    : m_MaxWorkers(std::max(size_t{1}, maxWorkers)), m_GetMemoryLoad(std::move(getMemoryLoad))
{
}

FractalSaveThreadPool::~FractalSaveThreadPool() { Shutdown(); }

FractalSaveThreadPool::GpuEncodingLease::GpuEncodingLease(FractalSaveThreadPool &pool) : m_Pool(pool)
{
    std::unique_lock lock(m_Pool.m_Mutex);
    // Accepted jobs must still acquire the encoder while Shutdown waits for them to finish.
    m_Pool.m_SlotAvailable.wait(lock, [this] { return !m_Pool.m_GpuEncodingBusy; });
    m_Pool.m_GpuEncodingBusy = true;
}

FractalSaveThreadPool::GpuEncodingLease::~GpuEncodingLease()
{
    {
        std::lock_guard lock(m_Pool.m_Mutex);
        m_Pool.m_GpuEncodingBusy = false;
    }
    // Capacity, completion, and encoder waiters share this condition variable. Wake all so
    // each waiter can recheck its own predicate, including during exception unwinding.
    m_Pool.m_SlotAvailable.notify_all();
}

FractalSaveThreadPool::GpuEncodingLease
FractalSaveThreadPool::AcquireGpuEncoding()
{
    return GpuEncodingLease(*this);
}

void
FractalSaveThreadPool::Submit(const TaskFactory &makeTask)
{
    {
        std::unique_lock lock(m_Mutex);
        while (m_Accepting &&
               (m_Outstanding == m_MaxWorkers || (m_Outstanding != 0 && m_GetMemoryLoad() > 90))) {
            m_SlotAvailable.wait_for(lock, std::chrono::milliseconds(100));
        }
        if (!m_Accepting) {
            throw std::runtime_error("save pool is shutting down");
        }
        ++m_Outstanding;
    }

    OutstandingReservation reservation(m_Mutex, m_SlotAvailable, m_Outstanding);
    Task task = makeTask();
    {
        std::lock_guard lock(m_Mutex);
        if (m_Workers.size() < m_Outstanding) {
            m_Workers.emplace_back(&FractalSaveThreadPool::WorkerLoop, this);
        }
        m_Queue.emplace_back(std::move(task));
        reservation.Commit();
    }
    m_WorkAvailable.notify_one();
}

bool
FractalSaveThreadPool::Cleanup(bool all)
{
    std::unique_lock lock(m_Mutex);
    const bool hadWork = m_Outstanding != 0 || m_UnreportedCompletions != 0;
    if (all) {
        m_SlotAvailable.wait(lock, [this] { return m_Outstanding == 0; });
    }
    const bool completed = m_UnreportedCompletions != 0;
    m_UnreportedCompletions = 0;
    return all ? hadWork : completed;
}

void
FractalSaveThreadPool::Shutdown()
{
    {
        std::unique_lock lock(m_Mutex);
        if (m_Stopping) {
            return;
        }
        m_Accepting = false;
        m_SlotAvailable.wait(lock, [this] { return m_Outstanding == 0; });
        m_Stopping = true;
    }
    m_SlotAvailable.notify_all();
    m_WorkAvailable.notify_all();
    for (auto &worker : m_Workers) {
        worker.join();
    }
    m_Workers.clear();
}

void
FractalSaveThreadPool::WorkerLoop()
{
    Environment::SetCurrentThreadName(L"Fractal save worker");
    for (;;) {
        {
            Task task;
            {
                std::unique_lock lock(m_Mutex);
                m_WorkAvailable.wait(lock, [this] { return m_Stopping || !m_Queue.empty(); });
                if (m_Queue.empty()) {
                    return;
                }
                task = std::move(m_Queue.front());
                m_Queue.pop_front();
            }
            try {
                task();
            } catch (const std::exception &ex) {
                FractalSharkLog::WriteException("Fractal save worker failed", ex, __FILE__, __LINE__);
            } catch (...) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Fractal save worker failed with an unknown exception";
            }
        }
        {
            std::lock_guard lock(m_Mutex);
            --m_Outstanding;
            ++m_UnreportedCompletions;
        }
        m_SlotAvailable.notify_all();
    }
}
