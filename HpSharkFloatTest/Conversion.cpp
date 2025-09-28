#include "TestVerbose.h"

#include <cuda_runtime.h>

#include "BenchmarkTimer.h"
#include "HpSharkFloat.cuh"
#include "TestTracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <assert.h>

static TestTracker Tests;

template <class SharkFloatParams>
void
TestConvertNumber(int testNum, const char *numberStr)
{
    using SF = HpSharkFloat<SharkFloatParams>;
    mpf_set_default_prec(SF::DefaultMpirBits);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\n\n=== Test " << testNum
                  << ": Convert number string to HpSharkFloat and back ===\n";
    }

    // ---------------- Lambdas ----------------
    auto mpf_init_prec = [&](mpf_t &v) { mpf_init2(v, SF::DefaultMpirBits); };

    // set v = 2^k  (k can be negative)
    auto set_two_pow = [&](mpf_t v, long k) {
        mpf_set_ui(v, 1);
        if (k >= 0)
            mpf_mul_2exp(v, v, static_cast<unsigned long>(k));
        else
            mpf_div_2exp(v, v, static_cast<unsigned long>(-k));
    };

    auto compare_within_eps =
        [&](const mpf_t a, const mpf_t b, const mpf_t eps, const char *label) -> bool {
        mpf_t diff, adiff;
        mpf_init_prec(diff);
        mpf_init_prec(adiff);
        mpf_sub(diff, a, b);
        mpf_abs(adiff, diff);

        const bool ok = (mpf_cmp(adiff, eps) <= 0);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\n[" << label
                      << "] |delta| = " << MpfToString<SharkFloatParams>(adiff, SF::DefaultPrecBits)
                      << "  <=?  eps = " << MpfToString<SharkFloatParams>(eps, SF::DefaultPrecBits)
                      << (ok ? "  [OK]\n" : "  [FAIL]\n");
        }

        mpf_clear(adiff);
        mpf_clear(diff);
        return ok;
    };

    auto set_pow2_minus_k = [&](mpf_t v, unsigned k) {
        mpf_set_ui(v, 1);
        mpf_div_2exp(v, v, k); // v = 2^{-k}
    };

    // ---------------- Parse input ----------------
    mpf_t mpf_x;
    mpf_init_prec(mpf_x);
    if (mpf_set_str(mpf_x, numberStr, 10) == -1) {
        std::cout << "Error setting mpf_x from input string\n";
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Original input values:\n";
        std::cout << "  numberStr: " << numberStr << "\n";
        std::cout << "  X (mpf): " << MpfToString<SharkFloatParams>(mpf_x, SF::DefaultPrecBits) << "\n";
        std::cout << "  X hex  : " << MpfToHexString(mpf_x) << "\n";
    }

    // ---------------- Build HpSharkFloat from mpf ----------------
    SF x_num{};
    x_num.MpfToHpGpu(mpf_x, SF::DefaultPrecBits);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nHighPrecisionNumber representations:\n";
        std::cout << "  X: " << x_num.ToString() << "\n";
        std::cout << "  X hex: " << x_num.ToHexString() << "\n";
    }

    long e2 = 0;
    if (mpf_sgn(mpf_x) != 0) {
        (void)mpf_get_d_2exp(&e2, mpf_x); // |x| ≈ m * 2^e2, m in [0.5, 1)
    } else {
        // For zero, fall back to a small absolute epsilon based on precision
        e2 = 0; // we'll still build eps from p below
    }

    // ---------------- Convert HpSharkFloat back to mpf (full path) ----------------
    mpf_t mpf_x_gpu;
    mpf_init_prec(mpf_x_gpu);
    x_num.HpGpuToMpf(mpf_x_gpu);

    // ---------------- HDR round-trips ----------------
    const auto hdrFloat = x_num.template ToHDRFloat<float>();
    const auto hdrDouble = x_num.template ToHDRFloat<double>();

    SF from_hdr_f, from_hdr_d;
    from_hdr_f.FromHDRFloat(hdrFloat);
    from_hdr_d.FromHDRFloat(hdrDouble);

    mpf_t mpf_x_hdrf;
    mpf_init_prec(mpf_x_hdrf);
    mpf_t mpf_x_hdrd;
    mpf_init_prec(mpf_x_hdrd);
    from_hdr_f.HpGpuToMpf(mpf_x_hdrf);
    from_hdr_d.HpGpuToMpf(mpf_x_hdrd);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nHDRFloat representations:\n";
        std::cout << "  HDRFloat<float>  : " << hdrFloat.template ToString<false>() << "\n";
        std::cout << "  HDRFloat<double> : " << hdrDouble.template ToString<false>() << "\n";

        std::cout << "\nRound-trip via HDR -> HpSharkFloat -> mpf:\n";
        std::cout << "  from HDR<float>  : "
                  << MpfToString<SharkFloatParams>(mpf_x_hdrf, SF::DefaultPrecBits) << "\n";
        std::cout << "  from HDR<double> : "
                  << MpfToString<SharkFloatParams>(mpf_x_hdrd, SF::DefaultPrecBits) << "\n";
    }

    // ---------------- Epsilons based on magnitude (ULP-scaled) ----------------
    constexpr long P_full = static_cast<long>(SF::DefaultMpirBits - 1); // your working precision
    constexpr long P_f = 23;                                            // float fraction bits
    constexpr long P_d = 52;                                            // double fraction bits

    mpf_t eps_full;
    mpf_init_prec(eps_full);
    mpf_t eps_hdrf;
    mpf_init_prec(eps_hdrf);
    mpf_t eps_hdrd;
    mpf_init_prec(eps_hdrd);

    // Use ~1 ULP for each path. For stricter 0.5 ULP, subtract 1 from the exponent below.
    set_two_pow(eps_full, e2 - P_full);
    set_two_pow(eps_hdrf, e2 - P_f);
    set_two_pow(eps_hdrd, e2 - P_d);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nEpsilons (ULP-scaled at |x|):\n";
        std::cout << "  e2(|x|)   : " << e2 << "\n";
        std::cout << "  eps_full  : " << MpfToString<SharkFloatParams>(eps_full, SF::DefaultPrecBits)
                  << "  (~1 ULP @ " << P_full << " bits)\n";
        std::cout << "  eps_hdrf  : " << MpfToString<SharkFloatParams>(eps_hdrf, SF::DefaultPrecBits)
                  << "  (~1 ULP @ float)\n";
        std::cout << "  eps_hdrd  : " << MpfToString<SharkFloatParams>(eps_hdrd, SF::DefaultPrecBits)
                  << "  (~1 ULP @ double)\n";
    }

    // ---------------- Three independent absolute-error checks ----------------
    const bool ok_full = compare_within_eps(mpf_x, mpf_x_gpu, eps_full, "conversion/full");
    const bool ok_hdr_f = compare_within_eps(mpf_x, mpf_x_hdrf, eps_hdrf, "conversion/hdr_float");
    const bool ok_hdr_d = compare_within_eps(mpf_x, mpf_x_hdrd, eps_hdrd, "conversion/hdr_double");

    // ---------------- Mark results ----------------
    if (!ok_full) {
        Tests.MarkFailed(testNum,
                         "conversion/full",
                         "abs error exceeded eps",
                         MpfToString<SharkFloatParams>(eps_full, SF::DefaultPrecBits));
        assert(false);
    } else {
        Tests.MarkSuccess(testNum, "conversion/full");
    }

    if (!ok_hdr_f) {
        Tests.MarkFailed(testNum,
                         "conversion/hdr_float",
                         "abs error exceeded eps",
                         MpfToString<SharkFloatParams>(eps_hdrf, SF::DefaultPrecBits));
        assert(false);
    } else {
        Tests.MarkSuccess(testNum, "conversion/hdr_float");
    }

    if (!ok_hdr_d) {
        Tests.MarkFailed(testNum,
                         "conversion/hdr_double",
                         "abs error exceeded eps",
                         MpfToString<SharkFloatParams>(eps_hdrd, SF::DefaultPrecBits));
        assert(false);
    } else {
        Tests.MarkSuccess(testNum, "conversion/hdr_double");
    }

    // ---------------- Cleanup ----------------
    mpf_clear(eps_hdrd);
    mpf_clear(eps_hdrf);
    mpf_clear(eps_full);
    mpf_clear(mpf_x_hdrd);
    mpf_clear(mpf_x_hdrf);
    mpf_clear(mpf_x_gpu);
    mpf_clear(mpf_x);
}

template <class SharkFloatParams>
bool
TestConversion(int testBase)
{
    const auto set1 = testBase + 10;
    // TestConvertNumber<SharkFloatParams>(set1 + 1, "0.0");
    // TestConvertNumber<SharkFloatParams>(set1 + 2, "1.0");
    TestConvertNumber<SharkFloatParams>(set1 + 3, "2.0");
    TestConvertNumber<SharkFloatParams>(set1 + 4, "3.0");

    const auto set2 = testBase + 20;
    TestConvertNumber<SharkFloatParams>(set2 + 1, "0.1");
    TestConvertNumber<SharkFloatParams>(set2 + 2, "0.2");
    TestConvertNumber<SharkFloatParams>(set2 + 3, "0.3");
    TestConvertNumber<SharkFloatParams>(set2 + 4, "0.4");
    TestConvertNumber<SharkFloatParams>(set2 + 5, "0.5");
    TestConvertNumber<SharkFloatParams>(set2 + 6, "0.6");
    TestConvertNumber<SharkFloatParams>(set2 + 7, "0.7");
    TestConvertNumber<SharkFloatParams>(set2 + 8, "0.8");
    TestConvertNumber<SharkFloatParams>(set2 + 9, "0.9");

    const auto set3 = testBase + 30;
    TestConvertNumber<SharkFloatParams>(set3 + 1, "1e-50");
    TestConvertNumber<SharkFloatParams>(set3 + 2, "1e-100");
    TestConvertNumber<SharkFloatParams>(set3 + 3, "1e-150");
    TestConvertNumber<SharkFloatParams>(set3 + 4, "1e-500");
    TestConvertNumber<SharkFloatParams>(set3 + 5, "1e-1000");
    TestConvertNumber<SharkFloatParams>(set3 + 6, "-1e-50");
    TestConvertNumber<SharkFloatParams>(set3 + 7, "-1e-100");
    TestConvertNumber<SharkFloatParams>(set3 + 8, "-1e-150");
    TestConvertNumber<SharkFloatParams>(set3 + 9, "-1e-500");

    const auto set4 = testBase + 40;
    TestConvertNumber<SharkFloatParams>(set4 + 1, "-1");
    TestConvertNumber<SharkFloatParams>(set4 + 2, "-2");
    TestConvertNumber<SharkFloatParams>(set4 + 3, "-3");
    TestConvertNumber<SharkFloatParams>(set4 + 4, "-4");

    const auto set5 = testBase + 50;
    TestConvertNumber<SharkFloatParams>(set5 + 1, "-0.1");
    TestConvertNumber<SharkFloatParams>(set5 + 2, "-0.2");
    TestConvertNumber<SharkFloatParams>(set5 + 3, "-0.3");
    TestConvertNumber<SharkFloatParams>(set5 + 4, "-0.4");
    TestConvertNumber<SharkFloatParams>(set5 + 5, "-0.5");
    TestConvertNumber<SharkFloatParams>(set5 + 6, "-0.6");
    TestConvertNumber<SharkFloatParams>(set5 + 7, "-0.7");
    TestConvertNumber<SharkFloatParams>(set5 + 8, "-0.8");
    TestConvertNumber<SharkFloatParams>(set5 + 9, "-0.9");
    TestConvertNumber<SharkFloatParams>(set5 + 10, "1.999999999");
    TestConvertNumber<SharkFloatParams>(set5 + 11, "1.99999999999999999999999999999");
    TestConvertNumber<SharkFloatParams>(set5 + 12, "1.9999999999999999999999999999999999999999999999");

    const auto set6 = testBase + 70;
    TestConvertNumber<SharkFloatParams>(set6 + 1, "4294967297");
    TestConvertNumber<SharkFloatParams>(set6 + 2, "18446744073709551617");
    TestConvertNumber<SharkFloatParams>(set6 + 3, "55340232221128654849"); // 2^65 + 2^64 + 1
    TestConvertNumber<SharkFloatParams>(set6 + 4, "-4294967297");
    TestConvertNumber<SharkFloatParams>(set6 + 5, "-18446744073709551617");
    TestConvertNumber<SharkFloatParams>(set6 + 6, "-55340232221128654849");
    TestConvertNumber<SharkFloatParams>(set6 + 7, "18446744073709551615");

    const auto set7 = testBase + 80;
    TestConvertNumber<SharkFloatParams>(set7 + 1, "4294967297.0000152587890625"); // 2^32 + 1 + 1/2^16
    TestConvertNumber<SharkFloatParams>(set7 + 2, "18446744073709551617.0000152587890625");
    TestConvertNumber<SharkFloatParams>(set7 + 3, "55340232221128654849.0000152587890625");

    return Tests.CheckAllTestsPassed();
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template bool TestConversion<SharkFloatParams>(int testBase);

ExplicitInstantiateAll();
