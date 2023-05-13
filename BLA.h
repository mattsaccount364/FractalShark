#pragma once

#include <complex>

class BLA {
public:

    double r2;
    double Ax;
    double Ay;
    double Bx;
    double By;
    int l;

    using Complex = std::complex<double>;

    BLA() = default;

    BLA(double r2, Complex A, Complex B, int l) {
        this->Ax = A.real();
        this->Ay = A.imag();
        this->Bx = B.real();
        this->By = B.imag();
        this->r2 = r2;
        this->l = l;
    }

    Complex getValue(Complex DeltaSubN, Complex DeltaSub0) {

        //double zxn = Ax * zx - Ay * zy + Bx * cx - By * cy;
        //double zyn = Ax * zy + Ay * zx + Bx * cy + By * cx;
        double zx = DeltaSubN.real();
        double zy = DeltaSubN.imag();
        double cx = DeltaSub0.real();
        double cy = DeltaSub0.imag();
        return Complex(Ax * zx - Ay * zy + Bx * cx - By * cy,
                       Ax * zy + Ay * zx + Bx * cy + By * cx);
        //return  Complex.AtXpBtY(A, DeltaSubN, B, DeltaSub0);
    }

    Complex getValue(Complex DeltaSubN, double cx) {
        double zx = DeltaSubN.real();
        double zy = DeltaSubN.imag();
        return Complex(Ax * zx - Ay * zy + Bx * cx,
                       Ax * zy + Ay * zx + By * cx);
    }

    Complex getValue(Complex DeltaSubN) {
        double zx = DeltaSubN.real();
        double zy = DeltaSubN.imag();
        return Complex(Ax * zx - Ay * zy,
                       Ax * zy + Ay * zx);
    }

    double hypotA() {
        return sqrt(Ax * Ax + Ay * Ay);
    }

    double hypotB() {
        return sqrt(Bx * Bx + By * By);
    }

    double getAx() {
        return Ax;
    }

    double getAy() {
        return Ay;
    }

    double getBx() {
        return Bx;
    }

    double getBy() {
        return By;
    }

    static BLA getGenericStep(double r2, Complex A, Complex B, int l) {
        return BLA(r2, A, B, l);
    }

    // A = y.A * x.A
    static Complex getNewA(BLA x, BLA y) {
        return Complex(y.Ax * x.Ax - y.Ay * x.Ay,
                       y.Ax * x.Ay + y.Ay * x.Ax);
    }

    // B = y.A * x.B + y.B
    static Complex getNewB(BLA x, BLA y) {
        double xBx = x.getBx();
        double xBy = x.getBy();
        return Complex(y.Ax * xBx - y.Ay * xBy + y.getBx(),
                       y.Ax * xBy + y.Ay * xBx + y.getBy());
    }

    int getL() {
        return l;
    }
};
