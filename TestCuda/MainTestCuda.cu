#include "Add.cuh"
#include "NullKernel.cuh"
#include "Conversion.cuh"
#include "HpGpu.cuh"
#include "Tests.cuh"

#include <iostream>
#include <mpir.h>
#include <conio.h>

// Function to perform the calculation on the host using MPIR
void computeNextXY_host(mpf_t x, mpf_t y, mpf_t a, mpf_t b, int num_iter) {
    mpf_t x_squared, y_squared, two_xy, temp_x, temp_y;
    mpf_init(x_squared);
    mpf_init(y_squared);
    mpf_init(two_xy);
    mpf_init(temp_x);
    mpf_init(temp_y);

    for (int iter = 0; iter < num_iter; ++iter) {
        mpf_mul(x_squared, x, x); // x^2
        mpf_mul(y_squared, y, y); // y^2
        mpf_mul(temp_y, x, y);    // xy
        mpf_mul_ui(two_xy, temp_y, 2); // 2xy

        mpf_sub(temp_x, x_squared, y_squared); // x^2 - y^2
        mpf_add(temp_x, temp_x, a);            // x^2 - y^2 + a
        mpf_add(temp_y, two_xy, b);            // 2xy + b

        mpf_set(x, temp_x);
        mpf_set(y, temp_y);
    }

    mpf_clear(x_squared);
    mpf_clear(y_squared);
    mpf_clear(two_xy);
    mpf_clear(temp_x);
    mpf_clear(temp_y);
}

void PressKey() {
    // Press any key to continue (win32)
    // on console, don't require a newline
    std::cout << "Press any key to continue...";
    _getch();
}

int main(int argc, char *argv[]) {
    if constexpr (!SkipCorrectnessTests) {
        //TestNullKernel();
        //PressKey();

        int testBase = 0;
        bool res = false;
        res = TestConversion(0);
        if (!res) {
            PressKey();
        }

        testBase = 200;
        res = TestAllBinaryOp<Operator::Multiply>(testBase);
        if (!res) {
            PressKey();
        }

        testBase = 400;
        res = TestAllBinaryOp<Operator::Add>(testBase);
        if (!res) {
            PressKey();
        }
    }

    //testBase = 600;
    //bool res = TestBinaryOperatorPerf<Operator::Add>(testBase);
    //if (!res) {
    //    PressKey();
    //}

    return 0;
}
