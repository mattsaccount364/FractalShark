#pragma once

#include <cstdlib>

#ifdef _MSC_VER
#include <malloc.h>

inline void *
fs_aligned_malloc(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

inline void
fs_aligned_free(void *ptr)
{
    _aligned_free(ptr);
}

#else

inline void *
fs_aligned_malloc(size_t size, size_t alignment)
{
    // C11 aligned_alloc requires size to be a multiple of alignment.
    size_t rounded = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, rounded);
}

inline void
fs_aligned_free(void *ptr)
{
    free(ptr);
}

#endif
