#pragma once

#include "CudaCrap.h"

#include <array>
#include <cassert>
#include <string>

enum class ThreeWayLargestOrdering : int {
    A_GT_AllOthers = 0,
    B_GT_AllOthers = 1,
    C_GT_AllOthers = 2,
    COUNT
};

class ThreeWayLargestInfo {
public:
    struct Info {
        ThreeWayLargestOrdering ordering;
        const char *description;
        bool normalizeA;
        bool normalizeB;
        bool normalizeC;
    };

    // ——— Three constexpr entries ———
    static constexpr Info A_GT_AllOthers{ThreeWayLargestOrdering::A_GT_AllOthers,
                                         "A is the largest",
                                         true, // normalize A against the others
                                         false,
                                         false};

    static constexpr Info B_GT_AllOthers{
        ThreeWayLargestOrdering::B_GT_AllOthers, "B is the largest", false, true, false};

    static constexpr Info C_GT_AllOthers{
        ThreeWayLargestOrdering::C_GT_AllOthers, "C is the largest", false, false, true};

    // ——— Public API ———

    static std::string
    ToString(ThreeWayLargestOrdering o)
    {
        return InfoFor(o).description;
    }

    static void
    OrderingToNormalize(ThreeWayLargestOrdering o, bool &useA, bool &useB, bool &useC)
    {
        auto &I = InfoFor(o);
        useA = I.normalizeA;
        useB = I.normalizeB;
        useC = I.normalizeC;
    }

private:
    // lookup table
    static constexpr std::array<Info, (int)ThreeWayLargestOrdering::COUNT> Table = {
        {A_GT_AllOthers, B_GT_AllOthers, C_GT_AllOthers}};

    static constexpr const Info &
    InfoFor(ThreeWayLargestOrdering o)
    {
        int idx = static_cast<int>(o);
        assert(idx >= 0 && idx < (int)ThreeWayLargestOrdering::COUNT);
        return Table[idx];
    }
};

//////////////////////////////////////////////////////////////////////////////

enum class ThreeWayMagnitudeOrdering : int {
    A_GT_B_GT_C = 0,
    A_GT_C_GT_B = 1,
    B_GT_A_GT_C = 2,
    B_GT_C_GT_A = 3,
    C_GT_A_GT_B = 4,
    C_GT_B_GT_A = 5,
    COUNT
};

class ThreeWayMagnitude {
public:
    struct Info {
        ThreeWayMagnitudeOrdering Ordering;
        const char *Description;
        bool UseNormalizeA;
        bool UseNormalizeB;
        bool UseNormalizeC;
        const char *XStr;
        const char *YStr;
        const char *ZStr;
    };

    // ——— Six constexpr entries, exactly matching the old structs ———
    static constexpr Info A_GT_B_GT_C{
        ThreeWayMagnitudeOrdering::A_GT_B_GT_C, "A > B > C", true, false, false, "A", "B", "C"};

    static constexpr Info A_GT_C_GT_B{
        ThreeWayMagnitudeOrdering::A_GT_C_GT_B, "A > C > B", true, false, false, "A", "C", "B"};

    static constexpr Info B_GT_A_GT_C{
        ThreeWayMagnitudeOrdering::B_GT_A_GT_C, "B > A > C", false, true, false, "B", "A", "C"};

    static constexpr Info B_GT_C_GT_A{
        ThreeWayMagnitudeOrdering::B_GT_C_GT_A, "B > C > A", false, true, false, "B", "C", "A"};

    static constexpr Info C_GT_A_GT_B{
        ThreeWayMagnitudeOrdering::C_GT_A_GT_B, "C > A > B", false, false, true, "C", "A", "B"};

    static constexpr Info C_GT_B_GT_A{
        ThreeWayMagnitudeOrdering::C_GT_B_GT_A, "C > B > A", false, false, true, "C", "B", "A"};

    // ——— Public API exactly as before ———

    static std::string
    ToStr(ThreeWayMagnitudeOrdering o)
    {
        return InfoFor(o).Description;
    }

    static void
    GetArrayName(std::string &x, std::string &y, std::string &z, ThreeWayMagnitudeOrdering o)
    {
        auto &I = InfoFor(o);
        x = I.XStr;
        y = I.YStr;
        z = I.ZStr;
    }

    static void
    OrderingToNormalize(ThreeWayMagnitudeOrdering o, bool &useA, bool &useB, bool &useC)
    {
        auto &I = InfoFor(o);
        useA = I.UseNormalizeA;
        useB = I.UseNormalizeB;
        useC = I.UseNormalizeC;
    }

private:
    // small constexpr table for lookup
    static constexpr std::array<Info, (int)ThreeWayMagnitudeOrdering::COUNT> Table = {
        {A_GT_B_GT_C, A_GT_C_GT_B, B_GT_A_GT_C, B_GT_C_GT_A, C_GT_A_GT_B, C_GT_B_GT_A}};

    static constexpr const Info &
    InfoFor(ThreeWayMagnitudeOrdering o)
    {
        int idx = static_cast<int>(o);
        assert(idx >= 0 && idx < (int)ThreeWayMagnitudeOrdering::COUNT);
        return Table[idx];
    }
};
