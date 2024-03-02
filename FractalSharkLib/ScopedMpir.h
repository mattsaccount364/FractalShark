#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

// Bump allocator that bumps downwards
struct ScopedMPIRAllocators {
    ScopedMPIRAllocators();
    ~ScopedMPIRAllocators();

    void InitScopedAllocators();

    static void InitTls();
    static void ShutdownTls();

private:
    void* (*ExistingMalloc) (size_t);
    void* (*ExistingRealloc) (void*, size_t, size_t);
    void (*ExistingFree) (void*, size_t);

    // It looks like we only need two blocks but we use four in case anything
    // changes in the RefOrbit calculation.  This all is pretty balanced together.
    static constexpr size_t NumBlocks = 4;
    static constexpr size_t BytesPerBlock = 256 * 1024;

    thread_local static uint8_t Allocated[NumBlocks][BytesPerBlock];
    thread_local static uint8_t* AllocatedEnd[NumBlocks];
    thread_local static size_t AllocatedIndex;
    thread_local static size_t AllocationsAndFrees[NumBlocks];

    // Currently, this one maxes out around 10.  So 10 live allocations at once.
    static std::atomic<size_t> MaxAllocatedDebug;

    static void* NewMalloc(size_t size);
    static void* NewRealloc(void* ptr, size_t old_size, size_t new_size);
    static void NewFree(void* ptr, size_t size);
};

struct ScopedMPIRPrecision
{
    unsigned saved_digits10;
    ScopedMPIRPrecision(unsigned digits10);
    ~ScopedMPIRPrecision();
    void reset(unsigned digits10);
    void reset();
};

struct ScopedMPIRPrecisionOptions
{
    boost::multiprecision::variable_precision_options saved_options;
    ScopedMPIRPrecisionOptions(boost::multiprecision::variable_precision_options opts);
    ~ScopedMPIRPrecisionOptions();
    void reset(boost::multiprecision::variable_precision_options opts);
};
