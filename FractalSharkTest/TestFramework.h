#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace TestFramework {

struct TestFailure {
    std::string file;
    int line;
    std::string message;
};

struct TestCase {
    std::string name;
    std::function<void()> func;
};

inline std::vector<TestCase> &
Registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

inline bool
Register(const char *name, std::function<void()> func)
{
    Registry().push_back({name, std::move(func)});
    return true;
}

[[noreturn]] inline void
Fail(const char *file, int line, const std::string &msg)
{
    TestFailure f;
    f.file = file;
    f.line = line;
    f.message = msg;
    throw f;
}

inline int
RunAllTests()
{
    int passed = 0;
    int failed = 0;
    const int total = static_cast<int>(Registry().size());

    std::cout << "Running " << total << " test(s)...\n\n";

    for (const auto &test : Registry()) {
        try {
            test.func();
            std::cout << "  PASS: " << test.name << "\n";
            ++passed;
        } catch (const TestFailure &e) {
            std::cerr << "  FAIL: " << test.name << "\n"
                      << "        " << e.file << ":" << e.line << " - " << e.message << "\n";
            ++failed;
        } catch (const std::exception &e) {
            std::cerr << "  FAIL: " << test.name << "\n"
                      << "        Unhandled exception: " << e.what() << "\n";
            ++failed;
        } catch (...) {
            std::cerr << "  FAIL: " << test.name << "\n"
                      << "        Unknown exception\n";
            ++failed;
        }
    }

    std::cout << "\n========================================\n"
              << passed << " passed, " << failed << " failed, " << total << " total\n";

    if (failed > 0) {
        std::cout << "RESULT: FAILED\n";
        return 1;
    }

    std::cout << "RESULT: PASSED\n";
    return 0;
}

} // namespace TestFramework

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

#define TEST(name)                                                                                 \
    static void test_##name();                                                                     \
    static bool reg_##name = TestFramework::Register(#name, test_##name);                          \
    static void test_##name()

#define ASSERT_TRUE(expr)                                                                          \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            TestFramework::Fail(__FILE__, __LINE__, "ASSERT_TRUE(" #expr ") failed");              \
        }                                                                                          \
    } while (0)

#define ASSERT_FALSE(expr)                                                                         \
    do {                                                                                           \
        if (expr) {                                                                                \
            TestFramework::Fail(__FILE__, __LINE__, "ASSERT_FALSE(" #expr ") failed");             \
        }                                                                                          \
    } while (0)

#define ASSERT_EQ(a, b)                                                                            \
    do {                                                                                           \
        const auto &tf_a_ = (a);                                                                   \
        const auto &tf_b_ = (b);                                                                   \
        if (!(tf_a_ == tf_b_)) {                                                                   \
            std::ostringstream tf_oss_;                                                             \
            tf_oss_ << "ASSERT_EQ(" #a ", " #b ") failed: " << tf_a_ << " != " << tf_b_;          \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                 \
        }                                                                                          \
    } while (0)

#define ASSERT_NE(a, b)                                                                            \
    do {                                                                                           \
        const auto &tf_a_ = (a);                                                                   \
        const auto &tf_b_ = (b);                                                                   \
        if (tf_a_ == tf_b_) {                                                                      \
            std::ostringstream tf_oss_;                                                             \
            tf_oss_ << "ASSERT_NE(" #a ", " #b ") failed: both equal " << tf_a_;                   \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                 \
        }                                                                                          \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                                     \
    do {                                                                                           \
        const auto tf_a_ = (a);                                                                    \
        const auto tf_b_ = (b);                                                                    \
        const auto tf_t_ = (tol);                                                                  \
        if (std::abs(tf_a_ - tf_b_) > tf_t_) {                                                    \
            std::ostringstream tf_oss_;                                                             \
            tf_oss_ << "ASSERT_NEAR(" #a ", " #b ", " #tol ") failed: |" << tf_a_ << " - "        \
                    << tf_b_ << "| = " << std::abs(tf_a_ - tf_b_) << " > " << tf_t_;              \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                 \
        }                                                                                          \
    } while (0)

#define ASSERT_THROWS(expr, extype)                                                                \
    do {                                                                                           \
        bool tf_caught_ = false;                                                                   \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (const extype &) {                                                                 \
            tf_caught_ = true;                                                                     \
        } catch (...) {                                                                            \
        }                                                                                          \
        if (!tf_caught_) {                                                                         \
            TestFramework::Fail(                                                                   \
                __FILE__, __LINE__,                                                                \
                "ASSERT_THROWS(" #expr ", " #extype ") - no " #extype " thrown");                  \
        }                                                                                          \
    } while (0)
