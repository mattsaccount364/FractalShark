// The custom heap allocator (HpSharkFloatLib/heap_allocator/HeapCpp.cpp) is
// Windows-only by design. On Linux all allocations flow through glibc malloc
// via Environment::SystemHeap*, so there is nothing for RegisterHeapCleanup()
// to clean up. This empty definition satisfies the link-time dependency and
// is a permanent Linux fixture, not a temporary workaround.
// Platform selection is handled in FractalSharkTest/CMakeLists.txt.

void
RegisterHeapCleanup()
{
}
