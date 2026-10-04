#include "TestFramework.h"

#include <stdexcept>

TEST(TestFramework_WildcardsMatchFullNames)
{
    ASSERT_TRUE(TestFramework::Matches("LA*", "LAEvaluation_ZeroBudget"));
    ASSERT_TRUE(TestFramework::Matches("*Zero?udget", "LAEvaluation_ZeroBudget"));
    ASSERT_TRUE(TestFramework::Matches("a**?c*", "abccc"));
    ASSERT_TRUE(TestFramework::Matches("*", ""));
    ASSERT_TRUE(TestFramework::Matches("Exact", "Exact"));
    ASSERT_FALSE(TestFramework::Matches("Exact", "ExactSuffix"));
    ASSERT_FALSE(TestFramework::Matches("LA*", "OtherLA"));
    ASSERT_FALSE(TestFramework::Matches("la*", "LAEvaluation"));
    ASSERT_FALSE(TestFramework::Matches("?", ""));
    ASSERT_FALSE(TestFramework::Matches("*abc", "ab"));
}

TEST(TestFramework_ArgumentParsingAndExclusionPrecedence)
{
    const char *arguments[] = {
        "tests", "--filter", "LA*", "--filter=HDR*", "--exclude=LA_Slow", "--list-tests", "--fail-fast"};
    TestFramework::RunOptions options;
    std::string error;
    ASSERT_TRUE(TestFramework::ParseArguments(7, arguments, options, error));
    ASSERT_TRUE(options.ListTests);
    ASSERT_TRUE(options.FailFast);
    ASSERT_TRUE(TestFramework::IsSelected("LA_Fast", options));
    ASSERT_TRUE(TestFramework::IsSelected("HDR_Fast", options));
    ASSERT_FALSE(TestFramework::IsSelected("LA_Slow", options));
    ASSERT_FALSE(TestFramework::IsSelected("Other", options));
    const char *invalid[] = {"tests", "--filter", "--list-tests"};
    ASSERT_FALSE(TestFramework::ParseArguments(3, invalid, options, error));
    const char *empty[] = {"tests", "--exclude="};
    ASSERT_FALSE(TestFramework::ParseArguments(2, empty, options, error));
    const char *unknown[] = {"tests", "--unknown"};
    ASSERT_FALSE(TestFramework::ParseArguments(2, unknown, options, error));
    const char *positional[] = {"tests", "LA*"};
    ASSERT_FALSE(TestFramework::ParseArguments(2, positional, options, error));
    const char *none[] = {"tests"};
    ASSERT_TRUE(TestFramework::ParseArguments(1, none, options, error));
    ASSERT_TRUE(TestFramework::IsSelected("Anything", options));
    const char *help[] = {"tests", "--help"};
    ASSERT_TRUE(TestFramework::ParseArguments(2, help, options, error));
    ASSERT_TRUE(options.Help);
}

TEST(TestFramework_ListingAndEmptySelectionsNeverExecute)
{
    int calls = 0;
    const std::vector<TestFramework::TestCase> tests = {{"LA_Fast", [&]() { ++calls; }},
                                                        {"Other", [&]() { ++calls; }}};
    TestFramework::RunOptions options;
    options.Filters = {"LA*"};
    options.ListTests = true;
    std::ostringstream output;
    std::ostringstream errors;
    ASSERT_EQ(TestFramework::RunTests(tests, options, output, errors), 0);
    ASSERT_EQ(output.str(), std::string{"LA_Fast\n"});
    ASSERT_EQ(calls, 0);
    options.Exclusions = {"*"};
    ASSERT_EQ(TestFramework::RunTests(tests, options, output, errors), 2);
    ASSERT_EQ(calls, 0);
    ASSERT_TRUE(errors.str().find("no tests selected") != std::string::npos);
}

TEST(TestFramework_FailFastAndExceptionBoundaries)
{
    int calls = 0;
    const std::vector<TestFramework::TestCase> tests = {
        {"Pass", [&]() { ++calls; }},
        {"Assert", []() { TestFramework::Fail("fixture", 1, "expected failure"); }},
        {"Exception", []() { throw std::runtime_error("expected exception"); }},
        {"Unknown", []() { throw 42; }},
        {"Last", [&]() { ++calls; }}};
    TestFramework::RunOptions options;
    options.FailFast = true;
    std::ostringstream output;
    std::ostringstream errors;
    ASSERT_EQ(TestFramework::RunTests(tests, options, output, errors), 1);
    ASSERT_EQ(calls, 1);
    ASSERT_TRUE(output.str().find("2 executed, 1 passed, 1 failed") != std::string::npos);
    ASSERT_TRUE(output.str().find("3 selected but unexecuted") != std::string::npos);
    options.FailFast = false;
    output.str("");
    errors.str("");
    calls = 0;
    ASSERT_EQ(TestFramework::RunTests(tests, options, output, errors), 1);
    ASSERT_EQ(calls, 2);
    ASSERT_TRUE(output.str().find("5 executed, 2 passed, 3 failed") != std::string::npos);
    ASSERT_TRUE(errors.str().find("expected exception") != std::string::npos);
    ASSERT_TRUE(errors.str().find("Unknown exception") != std::string::npos);
}
