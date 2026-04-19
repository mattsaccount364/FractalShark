// Linux stub for RegisterHeapCleanup(). On Windows the real implementation
// lives in HpSharkFloatLib/heap_allocator/HeapCpp.cpp; on Linux the test
// executable doesn't use the custom heap, so an empty stub is sufficient.
// Platform selection is handled in FractalSharkTest/CMakeLists.txt.

void
RegisterHeapCleanup()
{
}
