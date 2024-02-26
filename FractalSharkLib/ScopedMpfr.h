#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

struct scoped_mpfr_allocators {
    void* (*ExistingMalloc) (size_t);
    void* (*ExistingRealloc) (void*, size_t, size_t);
    void (*ExistingFree) (void*, size_t);

    scoped_mpfr_allocators();
    ~scoped_mpfr_allocators();

    static constexpr size_t NumBlocks = 8;
    static constexpr size_t BytesPerBlock = 1024 * 1024;

    thread_local static uint8_t Allocated[NumBlocks][BytesPerBlock];
    thread_local static size_t AllocatedSize[NumBlocks];
    thread_local static size_t AllocatedIndex;
    thread_local static size_t AllocationsAndFrees[NumBlocks];

private:
    static void* NewMalloc(size_t size);
    static void* NewRealloc(void* ptr, size_t old_size, size_t new_size);
    static void NewFree(void* ptr, size_t size);
};

struct scoped_mpfr_precision
{
    unsigned saved_digits10;
    scoped_mpfr_precision(unsigned digits10);
    ~scoped_mpfr_precision();
    void reset(unsigned digits10);
    void reset();
};

struct scoped_mpfr_precision_options
{
    boost::multiprecision::variable_precision_options saved_options;
    scoped_mpfr_precision_options(boost::multiprecision::variable_precision_options opts);
    ~scoped_mpfr_precision_options();
    void reset(boost::multiprecision::variable_precision_options opts);
};
