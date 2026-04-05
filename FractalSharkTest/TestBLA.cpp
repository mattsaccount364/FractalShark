#include "TestFramework.h"
#include "BLA.h"

#include <cmath>

// BLA<double> stores complex coefficients A=(Ax,Ay), B=(Bx,By), threshold r2, level l.
// getValue: Δz' = A·Δz + B·Δc (complex multiply-add)
// getNewA: A_new = y.A * x.A (complex multiply)
// getNewB: B_new = y.A * x.B + y.B (complex multiply + add)

using BLAd = BLA<double>;

// ---------------------------------------------------------------------------
// Construction & accessors
// ---------------------------------------------------------------------------

TEST(BLA_DefaultConstruction)
{
    BLAd b;
    ASSERT_NEAR(b.getR2(), 0.0, 1e-15);
    ASSERT_EQ(b.getL(), 0);
}

TEST(BLA_ParameterizedConstruction)
{
    // r2=4.0, A=(1,0), B=(0,1), level=3
    BLAd b(4.0, 1.0, 0.0, 0.0, 1.0, 3);
    ASSERT_NEAR(b.getR2(), 4.0, 1e-15);
    ASSERT_EQ(b.getL(), 3);
}

TEST(BLA_GetGenericStep)
{
    BLAd b = BLAd::getGenericStep(16.0, 2.0, 3.0, 4.0, 5.0, 7);
    ASSERT_NEAR(b.getR2(), 16.0, 1e-15);
    ASSERT_EQ(b.getL(), 7);
}

// ---------------------------------------------------------------------------
// getValue: Δz' = A·Δz + B·Δc
// ---------------------------------------------------------------------------

TEST(BLA_GetValue_Identity)
{
    // A = (1,0) (identity), B = (0,0) → Δz' = Δz
    BLAd b(4.0, 1.0, 0.0, 0.0, 0.0, 1);
    double dzR = 3.0, dzI = 4.0;
    double dc0R = 10.0, dc0I = 20.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 3.0, 1e-14);
    ASSERT_NEAR(dzI, 4.0, 1e-14);
}

TEST(BLA_GetValue_BOnly)
{
    // A = (0,0), B = (1,0) → Δz' = Δc
    BLAd b(4.0, 0.0, 0.0, 1.0, 0.0, 1);
    double dzR = 99.0, dzI = 99.0;
    double dc0R = 5.0, dc0I = 7.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 5.0, 1e-14);
    ASSERT_NEAR(dzI, 7.0, 1e-14);
}

TEST(BLA_GetValue_ComplexMultiply)
{
    // A = (1,2), B = (3,4), Δz = (1,0), Δc = (0,0)
    // Result = A·Δz = (1+2i)·(1+0i) = (1, 2)
    BLAd b(4.0, 1.0, 2.0, 3.0, 4.0, 1);
    double dzR = 1.0, dzI = 0.0;
    double dc0R = 0.0, dc0I = 0.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 1.0, 1e-14);
    ASSERT_NEAR(dzI, 2.0, 1e-14);
}

TEST(BLA_GetValue_Full)
{
    // A = (2,0), B = (0,3), Δz = (1,1), Δc = (1,0)
    // Re = 2*1 - 0*1 + 0*1 - 3*0 = 2
    // Im = 2*1 + 0*1 + 0*0 + 3*1 = 5
    BLAd b(4.0, 2.0, 0.0, 0.0, 3.0, 1);
    double dzR = 1.0, dzI = 1.0;
    double dc0R = 1.0, dc0I = 0.0;
    b.getValue(dzR, dzI, dc0R, dc0I);
    ASSERT_NEAR(dzR, 2.0, 1e-14);
    ASSERT_NEAR(dzI, 5.0, 1e-14);
}

// ---------------------------------------------------------------------------
// getNewA: A_new = y.A * x.A (complex multiply)
// ---------------------------------------------------------------------------

TEST(BLA_GetNewA)
{
    // x.A = (1,2), y.A = (3,4)
    // A_new = (3+4i)*(1+2i) = (3-8) + (6+4)i = (-5, 10)
    BLAd x(1.0, 1.0, 2.0, 0.0, 0.0, 1);
    BLAd y(1.0, 3.0, 4.0, 0.0, 0.0, 1);
    double realA, imagA;
    BLAd::getNewA(x, y, realA, imagA);
    ASSERT_NEAR(realA, -5.0, 1e-14);
    ASSERT_NEAR(imagA, 10.0, 1e-14);
}

// ---------------------------------------------------------------------------
// getNewB: B_new = y.A * x.B + y.B
// ---------------------------------------------------------------------------

TEST(BLA_GetNewB)
{
    // x.B = (1,0), y.A = (2,0), y.B = (10,20)
    // B_new = (2+0i)*(1+0i) + (10+20i) = (2+10, 0+20) = (12, 20)
    BLAd x(1.0, 0.0, 0.0, 1.0, 0.0, 1);
    BLAd y(1.0, 2.0, 0.0, 10.0, 20.0, 1);
    double realB, imagB;
    BLAd::getNewB(x, y, realB, imagB);
    ASSERT_NEAR(realB, 12.0, 1e-14);
    ASSERT_NEAR(imagB, 20.0, 1e-14);
}

TEST(BLA_GetNewB_Complex)
{
    // x.B = (1,1), y.A = (0,1), y.B = (0,0)
    // B_new = (0+1i)*(1+1i) + (0+0i) = (0-1) + (1+0)i = (-1, 1)
    BLAd x(1.0, 0.0, 0.0, 1.0, 1.0, 1);
    BLAd y(1.0, 0.0, 1.0, 0.0, 0.0, 1);
    double realB, imagB;
    BLAd::getNewB(x, y, realB, imagB);
    ASSERT_NEAR(realB, -1.0, 1e-14);
    ASSERT_NEAR(imagB, 1.0, 1e-14);
}

// ---------------------------------------------------------------------------
// hypotA / hypotB
// ---------------------------------------------------------------------------

TEST(BLA_HypotA)
{
    // A = (3,4) → |A| = 5
    BLAd b(1.0, 3.0, 4.0, 0.0, 0.0, 1);
    ASSERT_NEAR(b.hypotA(), 5.0, 1e-10);
}

TEST(BLA_HypotB)
{
    // B = (5,12) → |B| = 13
    BLAd b(1.0, 0.0, 0.0, 5.0, 12.0, 1);
    ASSERT_NEAR(b.hypotB(), 13.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Composition correctness:
// Composing two BLA steps (getNewA/getNewB) then applying getValue
// must match applying step1's getValue then step2's getValue sequentially.
// ---------------------------------------------------------------------------

TEST(BLA_CompositionMatchesSequential)
{
    // Two arbitrary BLA steps
    BLAd step1(4.0, 1.5, 0.5, 0.3, -0.2, 1); // A1=(1.5,0.5), B1=(0.3,-0.2)
    BLAd step2(4.0, 0.8, -0.6, 0.1, 0.4, 1);  // A2=(0.8,-0.6), B2=(0.1,0.4)

    // Input perturbation
    double dzR = 0.01, dzI = -0.02;
    double dc0R = 0.001, dc0I = 0.003;

    // Method 1: Apply sequentially
    double seqR = dzR, seqI = dzI;
    step1.getValue(seqR, seqI, dc0R, dc0I); // after step1
    step2.getValue(seqR, seqI, dc0R, dc0I); // after step2

    // Method 2: Compose steps, then apply once
    double compAr, compAi, compBr, compBi;
    BLAd::getNewA(step1, step2, compAr, compAi); // A_comp = step2.A * step1.A
    BLAd::getNewB(step1, step2, compBr, compBi); // B_comp = step2.A * step1.B + step2.B
    BLAd composed(4.0, compAr, compAi, compBr, compBi, 2);

    double compR = dzR, compI = dzI;
    composed.getValue(compR, compI, dc0R, dc0I);

    ASSERT_NEAR(compR, seqR, 1e-12);
    ASSERT_NEAR(compI, seqI, 1e-12);
}

// ---------------------------------------------------------------------------
// Perturbation theory: single-step BLA from reference orbit
//
// For Mandelbrot z → z² + c, the perturbation recurrence is:
//   δz_{n+1} = 2·z_n·δz_n + δz_n² + δc
// Dropping the δz² term (BLA linear approximation):
//   δz_{n+1} ≈ A·δz_n + B·δc   where A = 2·z_n, B = 1
// ---------------------------------------------------------------------------

TEST(BLA_PerturbationSingleStep)
{
    // Reference orbit for c = -1: z_0=0, z_1=-1, z_2=0, z_3=-1, ...
    // At n=1: z_1 = -1, so A = 2·(-1) = -2, B = 1
    // BLA at this step: A=(-2,0), B=(1,0)
    double z_n_re = -1.0, z_n_im = 0.0;
    BLAd step(256.0, 2.0 * z_n_re, 2.0 * z_n_im, 1.0, 0.0, 1);

    // Pixel perturbation: δc = (0.001, 0.002)
    double dcR = 0.001, dcI = 0.002;

    // After one reference orbit step: z_2 = z_1² + c = 1 + (-1) = 0
    // Direct perturbation (full, including δz² term):
    //   δz_1 = δc (since z_0=0, δz_0=0 → δz_1 = 2·0·0 + 0 + δc = δc)
    //   δz_2 = 2·z_1·δz_1 + δz_1² + δc
    //        = 2·(-1)·(0.001+0.002i) + (0.001+0.002i)² + (0.001+0.002i)
    //        = (-0.002-0.004i) + (-0.000003+0.000004i) + (0.001+0.002i)
    //        = (-0.001003, -0.001996)
    double dz1_re = dcR, dz1_im = dcI;
    // Direct (with quadratic term):
    double direct_re = 2.0 * z_n_re * dz1_re - 2.0 * z_n_im * dz1_im +
                       (dz1_re * dz1_re - dz1_im * dz1_im) + dcR;
    double direct_im = 2.0 * z_n_re * dz1_im + 2.0 * z_n_im * dz1_re +
                       2.0 * dz1_re * dz1_im + dcI;

    // BLA approximation (linear only):
    double blaR = dz1_re, blaI = dz1_im;
    step.getValue(blaR, blaI, dcR, dcI);

    // BLA should match direct to within O(δz²) ≈ 1e-5
    ASSERT_NEAR(blaR, direct_re, 1e-5);
    ASSERT_NEAR(blaI, direct_im, 1e-5);

    // BLA linear part alone: A·δz + B·δc = -2·δc + δc = -δc
    ASSERT_NEAR(blaR, -dcR, 1e-5);
    ASSERT_NEAR(blaI, -dcI, 1e-5);
}

// ---------------------------------------------------------------------------
// Multi-step: compose BLA steps from a short orbit, verify against
// direct sequential perturbation iteration.
//
// Reference orbit: c = -0.5 (inside main cardioid, doesn't escape)
//   z_0 = 0
//   z_1 = -0.5
//   z_2 = (-0.5)² + (-0.5) = -0.25
//   z_3 = (-0.25)² + (-0.5) = -0.4375
// ---------------------------------------------------------------------------

TEST(BLA_MultiStepOrbit)
{
    const double c_re = -0.5, c_im = 0.0;

    // Compute short reference orbit
    double z_re[4], z_im[4];
    z_re[0] = 0.0;       z_im[0] = 0.0;
    z_re[1] = c_re;      z_im[1] = c_im;
    z_re[2] = z_re[1] * z_re[1] - z_im[1] * z_im[1] + c_re;
    z_im[2] = 2.0 * z_re[1] * z_im[1] + c_im;
    z_re[3] = z_re[2] * z_re[2] - z_im[2] * z_im[2] + c_re;
    z_im[3] = 2.0 * z_re[2] * z_im[2] + c_im;

    // Build single-step BLAs: A_n = 2·z_n, B_n = 1
    BLAd s1(256.0, 2.0 * z_re[1], 2.0 * z_im[1], 1.0, 0.0, 1);
    BLAd s2(256.0, 2.0 * z_re[2], 2.0 * z_im[2], 1.0, 0.0, 1);
    BLAd s3(256.0, 2.0 * z_re[3], 2.0 * z_im[3], 1.0, 0.0, 1);

    // Compose all three into one multi-step BLA
    double compA12r, compA12i, compB12r, compB12i;
    BLAd::getNewA(s1, s2, compA12r, compA12i);
    BLAd::getNewB(s1, s2, compB12r, compB12i);
    BLAd comp12(256.0, compA12r, compA12i, compB12r, compB12i, 2);

    double compA123r, compA123i, compB123r, compB123i;
    BLAd::getNewA(comp12, s3, compA123r, compA123i);
    BLAd::getNewB(comp12, s3, compB123r, compB123i);
    BLAd comp123(256.0, compA123r, compA123i, compB123r, compB123i, 3);

    // Pixel perturbation
    double dcR = 0.0001, dcI = 0.0002;

    // Method A: Sequential application of single-step BLAs
    // δz_1 = δc (since δz_0 = 0 and z_0 = 0 → A_0 = 0, B_0 = 1... but we start at n=1)
    // Actually: δz_1 = δc, then apply s1, s2, s3 sequentially
    double seqR = dcR, seqI = dcI;
    s1.getValue(seqR, seqI, dcR, dcI);
    s2.getValue(seqR, seqI, dcR, dcI);
    s3.getValue(seqR, seqI, dcR, dcI);

    // Method B: Single application of composed 3-step BLA
    double multiR = dcR, multiI = dcI;
    comp123.getValue(multiR, multiI, dcR, dcI);

    // Should match exactly (composition is algebraically exact for linear maps)
    ASSERT_NEAR(multiR, seqR, 1e-12);
    ASSERT_NEAR(multiI, seqI, 1e-12);
}

// ---------------------------------------------------------------------------
// Short orbit loop: compute a reference orbit, build BLA steps, compose,
// then compare BLA-approximated perturbation against direct iteration
// for a nearby pixel. This is the end-to-end perturbation pipeline on CPU.
//
// Reference: c_ref = -0.75 (main cardioid boundary, attractive fixed point)
// Pixel:     c_pix = c_ref + δc, where δc = (1e-8, 1e-8)
// ---------------------------------------------------------------------------

TEST(BLA_OrbitLoopDirectComparison)
{
    const double c_re = -0.75, c_im = 0.0;
    const double dc_re = 1e-8, dc_im = 1e-8;
    const int orbitLen = 20;

    // --- Compute reference orbit z_n for c_ref ---
    double z_re[orbitLen + 1], z_im[orbitLen + 1];
    z_re[0] = 0.0;
    z_im[0] = 0.0;
    for (int n = 0; n < orbitLen; ++n) {
        double zr = z_re[n], zi = z_im[n];
        z_re[n + 1] = zr * zr - zi * zi + c_re;
        z_im[n + 1] = 2.0 * zr * zi + c_im;
    }

    // --- Direct perturbation: iterate δz for the nearby pixel ---
    // Full recurrence: δz_{n+1} = 2·z_n·δz_n + δz_n² + δc
    double dz_direct_re = 0.0, dz_direct_im = 0.0;
    for (int n = 0; n < orbitLen; ++n) {
        double zr = z_re[n], zi = z_im[n];
        double dr = dz_direct_re, di = dz_direct_im;
        // 2·z_n·δz_n (complex multiply)
        double lin_re = 2.0 * (zr * dr - zi * di);
        double lin_im = 2.0 * (zr * di + zi * dr);
        // δz_n² (complex square)
        double sq_re = dr * dr - di * di;
        double sq_im = 2.0 * dr * di;
        // δz_{n+1} = linear + quadratic + δc
        dz_direct_re = lin_re + sq_re + dc_re;
        dz_direct_im = lin_im + sq_im + dc_im;
    }

    // --- BLA approach: build single-step BLAs, compose, then apply ---
    // Single-step BLA at z_n: A = 2·z_n, B = 1
    // Compose all orbitLen steps into one multi-step BLA

    // Start with step at n=0: A=2·z_0=0, B=1 → this step just adds δc
    BLAd composed(256.0, 2.0 * z_re[0], 2.0 * z_im[0], 1.0, 0.0, 1);

    for (int n = 1; n < orbitLen; ++n) {
        BLAd step_n(256.0, 2.0 * z_re[n], 2.0 * z_im[n], 1.0, 0.0, 1);

        double newAr, newAi, newBr, newBi;
        BLAd::getNewA(composed, step_n, newAr, newAi);
        BLAd::getNewB(composed, step_n, newBr, newBi);
        composed = BLAd(256.0, newAr, newAi, newBr, newBi, n + 1);
    }

    // Apply composed BLA: δz_0 = 0, so we pass (0,0) as δz and (δc) as Δc₀
    double dz_bla_re = 0.0, dz_bla_im = 0.0;
    composed.getValue(dz_bla_re, dz_bla_im, dc_re, dc_im);

    // BLA is a linear approximation — it drops the δz² terms.
    // With δc = 1e-8, the quadratic terms are O(1e-16), so BLA and direct
    // should agree to ~1e-14 or better over 20 iterations.
    ASSERT_NEAR(dz_bla_re, dz_direct_re, 1e-10);
    ASSERT_NEAR(dz_bla_im, dz_direct_im, 1e-10);
}

