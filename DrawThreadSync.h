#pragma once

#include <memory>
#include <deque>
#include <thread>
#include <mutex>

struct DrawThreadSync {
    //DrawThreadSync& operator=(const DrawThreadSync&) = delete;
    //DrawThreadSync(const DrawThreadSync&) = delete;
    DrawThreadSync(
        size_t index,
        std::unique_ptr<std::thread> thread,
        std::deque<std::atomic_uint64_t>& draw_thread_atomics
    ) :
        m_Index(index),
        m_Thread(std::move(thread)),
        m_DrawThreadAtomics(draw_thread_atomics),
        m_DrawThreadReady{},
        m_DrawThreadProcessed{},
        m_TimeToExit{}
    {
    }

    size_t m_Index;
    std::mutex m_DrawThreadMutex;
    std::condition_variable m_DrawThreadCV;
    std::unique_ptr<std::thread> m_Thread;
    std::deque<std::atomic_uint64_t>& m_DrawThreadAtomics;
    bool m_DrawThreadReady;
    bool m_DrawThreadProcessed;
    bool m_TimeToExit;
};