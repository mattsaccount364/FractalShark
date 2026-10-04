#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
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
    bool RequiresGpu{};
    std::string DisabledReason;
    bool Golden{};
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

inline bool
RegisterCase(std::string name,
             std::function<void()> function,
             bool requiresGpu,
             std::string disabledReason,
             bool golden)
{
    Registry().push_back(
        {std::move(name), std::move(function), requiresGpu, std::move(disabledReason), golden});
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

struct RunOptions {
    std::vector<std::string> Filters;
    std::vector<std::string> Exclusions;
    bool ListTests{};
    bool FailFast{};
    bool Help{};
    bool UseGpu{};
    bool GenerateGoldens{};
    std::string OutputDirectory;
};

inline RunOptions &
CurrentOptions()
{
    static RunOptions options;
    return options;
}

class ScopedRunOptions {
public:
    explicit ScopedRunOptions(const RunOptions &options) : m_Previous{CurrentOptions()}
    {
        CurrentOptions() = options;
    }
    ~ScopedRunOptions() { CurrentOptions() = std::move(m_Previous); }

private:
    RunOptions m_Previous;
};

inline bool
Matches(std::string_view pattern, std::string_view name)
{
    size_t patternIndex = 0;
    size_t nameIndex = 0;
    size_t star = std::string_view::npos;
    size_t retry = 0;
    while (nameIndex < name.size()) {
        if (patternIndex < pattern.size() &&
            (pattern[patternIndex] == '?' || pattern[patternIndex] == name[nameIndex])) {
            ++patternIndex;
            ++nameIndex;
        } else if (patternIndex < pattern.size() && pattern[patternIndex] == '*') {
            star = patternIndex++;
            retry = nameIndex;
        } else if (star != std::string_view::npos) {
            patternIndex = star + 1;
            nameIndex = ++retry;
        } else {
            return false;
        }
    }
    while (patternIndex < pattern.size() && pattern[patternIndex] == '*') {
        ++patternIndex;
    }
    return patternIndex == pattern.size();
}

inline bool
IsSelected(std::string_view name, const RunOptions &options)
{
    bool included = options.Filters.empty();
    for (const auto &pattern : options.Filters) {
        included = included || Matches(pattern, name);
    }
    for (const auto &pattern : options.Exclusions) {
        if (Matches(pattern, name)) {
            return false;
        }
    }
    return included;
}

inline bool
ParseArguments(int argc, const char *const *argv, RunOptions &options, std::string &error)
{
    options = RunOptions{};
    error.clear();
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "--help") {
            options.Help = true;
        } else if (argument == "--list-tests") {
            options.ListTests = true;
        } else if (argument == "--fail-fast") {
            options.FailFast = true;
        } else if (argument == "--use-gpu") {
            options.UseGpu = true;
        } else if (argument == "--generate-goldens") {
            options.GenerateGoldens = true;
        } else {
            const auto equals = argument.find('=');
            const auto option = argument.substr(0, equals);
            if (option != "--filter" && option != "--exclude" && option != "--output-dir") {
                error = "unknown argument: " + std::string{argument};
                return false;
            }
            std::string_view value;
            if (equals != std::string_view::npos) {
                value = argument.substr(equals + 1);
            } else if (i + 1 < argc && !std::string_view{argv[i + 1]}.starts_with("--")) {
                value = argv[++i];
            }
            if (value.empty()) {
                error = std::string{option} + " requires a nonempty pattern";
                return false;
            }
            if (option == "--output-dir") {
                options.OutputDirectory = value;
            } else {
                auto &patterns = option == "--filter" ? options.Filters : options.Exclusions;
                patterns.emplace_back(value);
            }
        }
    }
    return true;
}

inline void
PrintHelp(std::ostream &output)
{
    output << "Usage: FractalSharkTest [--filter PATTERN] [--exclude PATTERN]\n"
              "                        [--list-tests] [--fail-fast] [--help]\n"
              "                        [--use-gpu] [--generate-goldens] [--output-dir PATH]\n"
              "GPU cases require --use-gpu; disabled incomplete cases never execute.\n"
              "Generation writes candidate CRCs and retains images; it does not validate goldens.\n"
              "Patterns match full names case-sensitively: * matches any sequence, ? one character.\n"
              "Repeat --filter to include groups; exclusions take precedence. No filters runs all.\n"
              "Examples:\n"
              "  FractalSharkTest --filter \"LA*\"\n"
              "  FractalSharkTest --exclude \"RenderGolden_*\"\n"
              "  FractalSharkTest --list-tests --filter \"*LA*\"\n";
}

inline int
RunTests(std::span<const TestCase> tests,
         const RunOptions &options,
         std::ostream &output,
         std::ostream &errors)
{
    ScopedRunOptions scopedOptions{options};
    size_t passed = 0;
    size_t failed = 0;
    size_t skipped = 0;
    size_t disabled = 0;
    size_t generated = 0;
    size_t selected = 0;
    for (const auto &test : tests) {
        selected += IsSelected(test.name, options);
    }
    if (selected == 0) {
        errors << "error: no tests selected (" << tests.size() << " registered)\n";
        return 2;
    }
    if (options.ListTests) {
        for (const auto &test : tests) {
            if (IsSelected(test.name, options)) {
                output << test.name;
                if (test.RequiresGpu) {
                    output << " [requires --use-gpu]";
                }
                if (!test.DisabledReason.empty()) {
                    output << " [disabled: " << test.DisabledReason << ']';
                }
                output << '\n';
            }
        }
        return 0;
    }

    output << "Running " << selected << " of " << tests.size() << " registered test(s)...\n\n";

    for (const auto &test : tests) {
        if (!IsSelected(test.name, options)) {
            continue;
        }
        if (!test.DisabledReason.empty()) {
            output << "  DISABLED: " << test.name << " - " << test.DisabledReason << '\n';
            ++disabled;
            continue;
        }
        if (test.RequiresGpu && !options.UseGpu) {
            output << "  SKIP: " << test.name << " - requires --use-gpu\n";
            ++skipped;
            continue;
        }
        try {
            test.func();
            if (test.Golden && options.GenerateGoldens) {
                output << "  GENERATED: " << test.name << '\n';
                ++generated;
            } else {
                output << "  PASS: " << test.name << "\n";
                ++passed;
            }
        } catch (const TestFailure &e) {
            errors << "  FAIL: " << test.name << "\n"
                   << "        " << e.file << ":" << e.line << " - " << e.message << "\n";
            ++failed;
        } catch (const std::exception &e) {
            errors << "  FAIL: " << test.name << "\n"
                   << "        Unhandled exception: " << e.what() << "\n";
            ++failed;
        } catch (...) {
            // Unknown exception types still fail only this test and keep the suite running.
            errors << "  FAIL: " << test.name << "\n"
                   << "        Unknown exception\n";
            ++failed;
        }
        if (options.FailFast && failed != 0) {
            break;
        }
    }

    output << "\n========================================\n"
           << tests.size() << " registered, " << selected << " selected, " << passed + failed + generated
           << " executed, " << passed << " passed, " << failed << " failed, "
           << selected - passed - failed - generated << " selected but unexecuted\n"
           << generated << " generated, " << skipped << " skipped, " << disabled << " disabled\n";

    if (failed > 0) {
        output << "RESULT: FAILED\n";
        return 1;
    }

    output << (options.GenerateGoldens ? "RESULT: GENERATION COMPLETED (goldens not validated)\n"
                                       : "RESULT: PASSED\n");
    return 0;
}

inline int
RunAllTests()
{
    return RunTests(Registry(), RunOptions{}, std::cout, std::cerr);
}

} // namespace TestFramework

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

#define TEST(name)                                                                                      \
    static void test_##name();                                                                          \
    static bool reg_##name = TestFramework::Register(#name, test_##name);                               \
    static void test_##name()

#define ASSERT_TRUE(expr)                                                                               \
    do {                                                                                                \
        if (!(expr)) {                                                                                  \
            TestFramework::Fail(__FILE__, __LINE__, "ASSERT_TRUE(" #expr ") failed");                   \
        }                                                                                               \
    } while (0)

#define ASSERT_FALSE(expr)                                                                              \
    do {                                                                                                \
        if (expr) {                                                                                     \
            TestFramework::Fail(__FILE__, __LINE__, "ASSERT_FALSE(" #expr ") failed");                  \
        }                                                                                               \
    } while (0)

#define ASSERT_EQ(a, b)                                                                                 \
    do {                                                                                                \
        const auto &tf_a_ = (a);                                                                        \
        const auto &tf_b_ = (b);                                                                        \
        if (!(tf_a_ == tf_b_)) {                                                                        \
            std::ostringstream tf_oss_;                                                                 \
            tf_oss_ << "ASSERT_EQ(" #a ", " #b ") failed: " << tf_a_ << " != " << tf_b_;                \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                     \
        }                                                                                               \
    } while (0)

#define ASSERT_NE(a, b)                                                                                 \
    do {                                                                                                \
        const auto &tf_a_ = (a);                                                                        \
        const auto &tf_b_ = (b);                                                                        \
        if (tf_a_ == tf_b_) {                                                                           \
            std::ostringstream tf_oss_;                                                                 \
            tf_oss_ << "ASSERT_NE(" #a ", " #b ") failed: both equal " << tf_a_;                        \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                     \
        }                                                                                               \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                                          \
    do {                                                                                                \
        const auto tf_a_ = (a);                                                                         \
        const auto tf_b_ = (b);                                                                         \
        const auto tf_t_ = (tol);                                                                       \
        if (std::abs(tf_a_ - tf_b_) > tf_t_) {                                                          \
            std::ostringstream tf_oss_;                                                                 \
            tf_oss_ << "ASSERT_NEAR(" #a ", " #b ", " #tol ") failed: |" << tf_a_ << " - " << tf_b_     \
                    << "| = " << std::abs(tf_a_ - tf_b_) << " > " << tf_t_;                             \
            TestFramework::Fail(__FILE__, __LINE__, tf_oss_.str());                                     \
        }                                                                                               \
    } while (0)

#define ASSERT_THROWS(expr, extype)                                                                     \
    do {                                                                                                \
        bool tf_caught_ = false;                                                                        \
        try {                                                                                           \
            (void)(expr);                                                                               \
        } catch (const extype &) {                                                                      \
            tf_caught_ = true;                                                                          \
        } catch (...) {                                                                                 \
            /* A different exception type is a failed ASSERT_THROWS expectation. */                     \
        }                                                                                               \
        if (!tf_caught_) {                                                                              \
            TestFramework::Fail(                                                                        \
                __FILE__, __LINE__, "ASSERT_THROWS(" #expr ", " #extype ") - no " #extype " thrown");   \
        }                                                                                               \
    } while (0)
