// Test for HDRFloat comparison with unreduced mantissas
// This test verifies that compareToBothPositiveReduced and compareToBothPositive
// correctly handle cases where mantissas are not in the normalized range [1.0, 2.0)

#include "../HpSharkFloatLib/HDRFloat.h"
#include <iostream>
#include <cassert>

void TestHDRFloatComparisons() {
    std::cout << "=== Testing HDRFloat Comparison Functions ===" << std::endl;
    
    int testsPassed = 0;
    int totalTests = 0;
    
    // Test 1: Equal values, one unreduced
    // A: 1.0 * 2^10 = 1024
    // B: 2.0 * 2^9 = 1024 (not reduced, should become 1.0 * 2^10)
    {
        totalTests++;
        HDRFloat<double> a(10, 1.0);
        HDRFloat<double> b(9, 2.0);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 0) {
            std::cout << "Test 1 PASSED: Equal values with unreduced operand" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 1 FAILED: Expected 0, got " << result << std::endl;
        }
    }
    
    // Test 2: A < B, B unreduced
    // A: 1.0 * 2^10 = 1024
    // B: 2.5 * 2^9 = 1280 (not reduced, should become 1.25 * 2^10)
    {
        totalTests++;
        HDRFloat<double> a(10, 1.0);
        HDRFloat<double> b(9, 2.5);
        int result = a.compareToBothPositiveReduced(b);
        if (result == -1) {
            std::cout << "Test 2 PASSED: A < B with unreduced operand" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 2 FAILED: Expected -1, got " << result << std::endl;
        }
    }
    
    // Test 3: A > B, A unreduced
    // A: 3.0 * 2^9 = 1536 (not reduced, should become 1.5 * 2^10)
    // B: 1.0 * 2^10 = 1024
    {
        totalTests++;
        HDRFloat<double> a(9, 3.0);
        HDRFloat<double> b(10, 1.0);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 1) {
            std::cout << "Test 3 PASSED: A > B with unreduced operand" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 3 FAILED: Expected 1, got " << result << std::endl;
        }
    }
    
    // Test 4: Both unreduced, equal
    // A: 4.0 * 2^8 = 1024 (not reduced, should become 1.0 * 2^10)
    // B: 2.0 * 2^9 = 1024 (not reduced, should become 1.0 * 2^10)
    {
        totalTests++;
        HDRFloat<double> a(8, 4.0);
        HDRFloat<double> b(9, 2.0);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 0) {
            std::cout << "Test 4 PASSED: Both unreduced, equal values" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 4 FAILED: Expected 0, got " << result << std::endl;
        }
    }
    
    // Test 5: Both properly reduced, normal comparison
    // A: 1.5 * 2^10 = 1536
    // B: 1.2 * 2^10 = 1228.8
    {
        totalTests++;
        HDRFloat<double> a(10, 1.5);
        HDRFloat<double> b(10, 1.2);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 1) {
            std::cout << "Test 5 PASSED: Both reduced, A > B" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 5 FAILED: Expected 1, got " << result << std::endl;
        }
    }
    
    // Test 6: compareToBothPositive should also work
    {
        totalTests++;
        HDRFloat<double> a(10, 1.0);
        HDRFloat<double> b(9, 2.0);
        int result = a.compareToBothPositive(b);
        if (result == 0) {
            std::cout << "Test 6 PASSED: compareToBothPositive works with unreduced" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 6 FAILED: Expected 0, got " << result << std::endl;
        }
    }
    
    // Test 7: Edge case with mantissa close to 2.0
    // A: 1.0 * 2^100
    // B: 1.99 * 2^99
    {
        totalTests++;
        HDRFloat<double> a(100, 1.0);
        HDRFloat<double> b(99, 1.99);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 1) {
            std::cout << "Test 7 PASSED: Edge case with mantissa close to 2.0" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 7 FAILED: Expected 1, got " << result << std::endl;
        }
    }
    
    // Test 8: Float type
    {
        totalTests++;
        HDRFloat<float> a(10, 1.0f);
        HDRFloat<float> b(9, 2.0f);
        int result = a.compareToBothPositiveReduced(b);
        if (result == 0) {
            std::cout << "Test 8 PASSED: Float type works correctly" << std::endl;
            testsPassed++;
        } else {
            std::cout << "Test 8 FAILED: Expected 0, got " << result << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== Test Summary: " << testsPassed << "/" << totalTests << " tests passed ===" << std::endl;
    
    if (testsPassed == totalTests) {
        std::cout << "SUCCESS: All HDRFloat comparison tests passed!" << std::endl;
    } else {
        std::cout << "FAILURE: " << (totalTests - testsPassed) << " test(s) failed!" << std::endl;
        exit(1);
    }
}

#ifndef NO_MAIN
int main() {
    TestHDRFloatComparisons();
    return 0;
}
#endif
