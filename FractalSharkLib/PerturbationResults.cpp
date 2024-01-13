#include "stdafx.h"
#include "PerturbationResults.h"

// Returns the current time as a string
std::wstring GetTimeAsString(size_t generation_number) {
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y-%m-%d-%H-%M-%S"); // HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    auto res = oss.str();

    std::wstring wide;
    std::transform(res.begin(), res.end(), std::back_inserter(wide), [](char c) {
        return (wchar_t)c;
        });

    wide += L"-" + std::to_wstring(generation_number);

    return wide;
}
