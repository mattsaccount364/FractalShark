template<typename IterType, class T>
__global__
void mandel_1x_float_perturb_scaled(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    MattPerturbSingleResults<IterType, float, PerturbExtras::Bad> PerturbFloat,
    MattPerturbSingleResults<IterType, T, PerturbExtras::Bad> PerturbDouble,
    int width,
    int height,
    T cx,
    T cy,
    T dx,
    T dy,
    T centerX,
    T centerY,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * Y + X;
    size_t idx = ConvertLocToIndex(X, Y, width);

    //if (OutputIterMatrix[idx] != 0) {
    //    return;
    //}

    IterType iter = 0;
    IterType RefIteration = 0;
    T DeltaReal = dx * X - centerX;
    HdrReduce(DeltaReal);

    T DeltaImaginary = -dy * Y - centerY;
    HdrReduce(DeltaImaginary);

    // DeltaSubNWX = 2 * DeltaSubNWX * PerturbFloat.x[RefIteration] - 2 * DeltaSubNWY * PerturbFloat.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * PerturbFloat.y[RefIteration] + 2 * DeltaSubNWY * PerturbFloat.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    T S = HdrSqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);
    HdrReduce(S);

    //double S = 1;
    float DeltaSub0DX = (float)(DeltaReal / S);
    float DeltaSub0DY = (float)(DeltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    float s = (float)S;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    IterType MaxRefIteration = PerturbFloat.size - 1;

    T TwoFiftySix = T(256.0);

    while (iter < n_iterations) {
        const MattReferenceSingleIter<float, PerturbExtras::Bad>* curFloatIter = &PerturbFloat.iters[RefIteration];
        const MattReferenceSingleIter<T, PerturbExtras::Bad>* curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curFloatIter->bad == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * curFloatIter->x * 2 -
                DeltaSubNWYOrig * curFloatIter->y * 2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y * 2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x * 2 +
                DeltaSub0DY;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const float tempZX =
                curFloatIter->x + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                curFloatIter->y + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const float DeltaSubNWXSquared = DeltaSubNWX * DeltaSubNWX;
            const float DeltaSubNWYSquared = DeltaSubNWY * DeltaSubNWY;
            const float w2 = DeltaSubNWXSquared + DeltaSubNWYSquared;
            const float normDeltaSubN = w2 * s * s;

            T DoubleTempZX;
            T DoubleTempZY;

            const bool zn_size_OK = (zn_size < 256.0f);
            const bool test1a = (zn_size < normDeltaSubN);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2 >= w2threshold) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            }
            else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + (T)DeltaSubNWX * S); // Xxrd, xr
                //HdrReduce(DoubleTempZX);
                DoubleTempZY = (curDoubleIter->y + (T)DeltaSubNWY * S); // Xxid, xi
                //HdrReduce(DoubleTempZY);

                RefIteration = 0;
                S = HdrSqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                HdrReduce(S);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = (T)DeltaSubNWX * S;
                //HdrReduce(DoubleTempZX);
                DoubleTempZY = (T)DeltaSubNWY * S;
                //HdrReduce(DoubleTempZY);

                S = HdrSqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                HdrReduce(S);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        }
        else {
            // Do full iteration at double precision
            T DeltaSubNWXOrig = (T)DeltaSubNWX;
            T DeltaSubNWYOrig = (T)DeltaSubNWY;

            T DoubleTempDeltaSubNWX = DeltaSubNWXOrig * curDoubleIter->x * 2;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX -= DeltaSubNWYOrig * curDoubleIter->y * 2;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX += S * DeltaSubNWXOrig * DeltaSubNWXOrig;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX -= S * DeltaSubNWYOrig * DeltaSubNWYOrig;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX += DeltaReal / S;
            HdrReduce(DoubleTempDeltaSubNWX);

            T DoubleTempDeltaSubNWY = DeltaSubNWXOrig * (curDoubleIter->y * 2 + T(2) * S * DeltaSubNWYOrig);
            //HdrReduce(DoubleTempDeltaSubNWY);
            DoubleTempDeltaSubNWY += DeltaSubNWYOrig * curDoubleIter->x * 2;
            //HdrReduce(DoubleTempDeltaSubNWY);
            DoubleTempDeltaSubNWY += DeltaImaginary / S;
            HdrReduce(DoubleTempDeltaSubNWY);

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const T tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const T tempZY =
                curDoubleIter->y +
                DoubleTempDeltaSubNWY * S; // Xxid

            T zn_size =
                tempZX * tempZX + tempZY * tempZY;
            HdrReduce(zn_size);

            if (!HdrCompareToBothPositiveReducedLT<T, 256>(zn_size)) {
                break;
            }

            const T TwoS = S * S;
            T normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * TwoS +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * TwoS;
            HdrReduce(normDeltaSubN);

            T DeltaSubNWXNew;
            T DeltaSubNWYNew;

            if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = HdrSqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            HdrReduce(S);
            s = (float)S;
            twos = 2 * s;

            DeltaSub0DX = (float)(DeltaReal / S);
            DeltaSub0DY = (float)(DeltaImaginary / S);
            DeltaSubNWX = (float)(DeltaSubNWXNew / S);
            DeltaSubNWY = (float)(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    OutputIterMatrix[idx] = iter;
}

template<typename IterType, int32_t LM2>
__global__
void mandel_1x_float_perturb_scaled_bla(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    MattPerturbSingleResults<IterType, float, PerturbExtras::Bad> PerturbFloat,
    MattPerturbSingleResults<IterType, double, PerturbExtras::Bad> PerturbDouble,
    GPU_BLAS<IterType, double, BLA<double>, LM2> doubleBlas,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * Y + X;
    size_t idx = ConvertLocToIndex(X, Y, width);

    //if (OutputIterMatrix[idx] != 0) {
    //    return;
    //}

    using GPUBLA_TYPE = BLA<double>;
    char __shared__ SharedMem[sizeof(SharedMemStruct<double>)];
    auto* shared =
        reinterpret_cast<SharedMemStruct<double>*>(SharedMem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        GPUBLA_TYPE** elts = doubleBlas.GetB();

        for (size_t i = 0; i < doubleBlas.m_NumLevels; i++) {
            shared->altB[i] = elts[i];
        }
    }

    __syncthreads();

    IterType iter = 0;
    IterType RefIteration = 0;
    const double DeltaReal = dx * X - centerX;
    const double DeltaImaginary = -dy * Y - centerY;
    double S = sqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);
    float DeltaSub0DX = (float)(DeltaReal / S);
    float DeltaSub0DY = (float)(DeltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    double DeltaNormSquared = 0;

    float s = (float)S;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    IterType MaxRefIteration = PerturbFloat.size - 1;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<float, PerturbExtras::Bad>* curFloatIter = &PerturbFloat.iters[RefIteration];
        const MattReferenceSingleIter<double, PerturbExtras::Bad>* curDoubleIter = &PerturbDouble.iters[RefIteration];

        double DeltaSubNX = DeltaSubNWX * S;
        double DeltaSubNY = DeltaSubNWY * S;

        const BLA<double>* b = nullptr;

        b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared);
        if (b != nullptr) {
            for (;;) {
                int l = b->getL();

                // TODO this first RefIteration + l check bugs me
                if (RefIteration + l >= PerturbDouble.size) {
                    break;
                }

                if (iter + l >= n_iterations) {
                    break;
                }

                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaReal, DeltaImaginary);

                curDoubleIter = &PerturbDouble.iters[RefIteration];
                const double tempZX = curDoubleIter->x + DeltaSubNX;
                const double tempZY = curDoubleIter->y + DeltaSubNY;
                const double normSquared = tempZX * tempZX + tempZY * tempZY;
                DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                if (normSquared > 256) {
                    break;
                }

                if (normSquared < DeltaNormSquared ||
                    RefIteration >= PerturbFloat.size - 1) {
                    DeltaSubNX = tempZX;
                    DeltaSubNY = tempZY;
                    DeltaNormSquared = normSquared;
                    RefIteration = 0;
                }

                b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared);
                if (b == nullptr) {
                    double DoubleTempZX = DeltaSubNX;
                    double DoubleTempZY = DeltaSubNY;

                    DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                    S = sqrt(DeltaNormSquared);
                    s = (float)S;
                    twos = 2 * s;

                    DeltaSub0DX = (float)(DeltaReal / S);
                    DeltaSub0DY = (float)(DeltaImaginary / S);
                    DeltaSubNWX = (float)(DoubleTempZX / S);
                    DeltaSubNWY = (float)(DoubleTempZY / S);
                    break;
                }
            }
        }

        if (iter >= n_iterations) {
            break;
        }

        curFloatIter = &PerturbFloat.iters[RefIteration];
        curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curFloatIter->bad == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * curFloatIter->x * 2 -
                DeltaSubNWYOrig * curFloatIter->y * 2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y * 2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x * 2 +
                DeltaSub0DY;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const float tempZX =
                curFloatIter->x + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                curFloatIter->y + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const float DeltaSubNWXSquared = DeltaSubNWX * DeltaSubNWX;
            const float DeltaSubNWYSquared = DeltaSubNWY * DeltaSubNWY;
            const float w2 = DeltaSubNWXSquared + DeltaSubNWYSquared;
            const float normDeltaSubN = w2 * s * s;
            DeltaNormSquared = normDeltaSubN; // Yay

            double DoubleTempZX;
            double DoubleTempZY;

            const bool zn_size_OK = (zn_size < 256.0f);
            const bool test1a = (zn_size < normDeltaSubN);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2 >= w2threshold) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            }
            else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + (double)DeltaSubNWX * S); // Xxrd, xr
                DoubleTempZY = (curDoubleIter->y + (double)DeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;

                DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                S = sqrt(DeltaNormSquared);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = DeltaSubNWX * S;
                DoubleTempZY = DeltaSubNWY * S;

                DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                S = sqrt(DeltaNormSquared);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        }
        else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = DeltaSubNWX;
            double DeltaSubNWYOrig = DeltaSubNWY;

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * curDoubleIter->x * 2 -
                DeltaSubNWYOrig * curDoubleIter->y * 2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y * 2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x * 2 +
                DeltaImaginary / S;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const double tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                curDoubleIter->y +
                DoubleTempDeltaSubNWY * S; // Xxid

            const double zn_size =
                tempZX * tempZX + tempZY * tempZY;

            if (zn_size > 256.0) {
                break;
            }

            const double TwoS = S * S;
            const double normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * TwoS +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * TwoS;
            DeltaNormSquared = normDeltaSubN; // Yay

            double DeltaSubNWXNew;
            double DeltaSubNWYNew;

            if (zn_size < normDeltaSubN ||
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            DeltaNormSquared = DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew; // Yay
            S = sqrt(DeltaNormSquared);
            s = (float)S;
            twos = 2 * s;

            DeltaSub0DX = (float)(DeltaReal / S);
            DeltaSub0DY = (float)(DeltaImaginary / S);
            DeltaSubNWX = (float)(DeltaSubNWXNew / S);
            DeltaSubNWY = (float)(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    OutputIterMatrix[idx] = iter;
}

template<typename IterType>
__global__
void mandel_2x_float_perturb_scaled(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    MattPerturbSingleResults<IterType, dblflt, PerturbExtras::Bad> PerturbDoubleFlt,
    MattPerturbSingleResults<IterType, double, PerturbExtras::Bad> PerturbDouble,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    //size_t idx = width * Y + X;
    size_t idx = ConvertLocToIndex(X, Y, width);

    //if (OutputIterMatrix[idx] != 0) {
    //    return;
    //}

    IterType iter = 0;
    IterType RefIteration = 0;
    const double DeltaReal = dx * X - centerX;
    const double DeltaImaginary = -dy * Y - centerY;

    // DeltaSubNWX = 2 * DeltaSubNWX * PerturbDoubleFlt.x[RefIteration] - 2 * DeltaSubNWY * PerturbDoubleFlt.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * PerturbDoubleFlt.y[RefIteration] + 2 * DeltaSubNWY * PerturbDoubleFlt.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    double S = sqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);

    //double S = 1;
    dblflt DeltaSub0DX = double_to_dblflt(DeltaReal / S);
    dblflt DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
    dblflt DeltaSubNWX = add_float_to_dblflt(0, 0);
    dblflt DeltaSubNWY = add_float_to_dblflt(0, 0);

    dblflt s = double_to_dblflt(S);
    dblflt twos = add_dblflt(s, s);
    const dblflt w2threshold = double_to_dblflt(exp(log(LARGE_MANTISSA) / 2));
    IterType MaxRefIteration = PerturbDoubleFlt.size - 1;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<dblflt, PerturbExtras::Bad>* curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
        const MattReferenceSingleIter<double, PerturbExtras::Bad>* curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curDblFloatIter->bad == false) {
            const dblflt DeltaSubNWXOrig = DeltaSubNWX;
            const dblflt DeltaSubNWYOrig = DeltaSubNWY;

            const dblflt Two = add_float_to_dblflt(0, 2);
            const dblflt tempX = mul_dblflt(curDblFloatIter->x, Two);
            const dblflt tempY = mul_dblflt(curDblFloatIter->y, Two);

            DeltaSubNWX = mul_dblflt(DeltaSubNWXOrig, tempX);
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(DeltaSubNWYOrig, tempY));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWXOrig), DeltaSubNWXOrig));
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWYOrig), DeltaSubNWYOrig));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, DeltaSub0DX);

            DeltaSubNWY = mul_dblflt(DeltaSubNWXOrig, (add_dblflt(tempY, mul_dblflt(twos, DeltaSubNWYOrig))));
            DeltaSubNWY = add_dblflt(DeltaSubNWY, mul_dblflt(DeltaSubNWYOrig, tempX));
            DeltaSubNWY = add_dblflt(DeltaSubNWY, DeltaSub0DY);

            ++RefIteration;
            curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const dblflt tempZX =
                add_dblflt(curDblFloatIter->x, mul_dblflt(DeltaSubNWX, s)); // Xxrd

            const dblflt tempZY =
                add_dblflt(curDblFloatIter->y, mul_dblflt(DeltaSubNWY, s)); // Xxid

            const dblflt zn_size =
                add_dblflt(sqr_dblflt(tempZX), sqr_dblflt(tempZY));

            const dblflt DeltaSubNWXSquared = sqr_dblflt(DeltaSubNWX);
            const dblflt DeltaSubNWYSquared = sqr_dblflt(DeltaSubNWY);
            const dblflt w2 = add_dblflt(DeltaSubNWXSquared, DeltaSubNWYSquared);
            const dblflt normDeltaSubN = mul_dblflt(w2, sqr_dblflt(s));

            double DoubleTempZX;
            double DoubleTempZY;

            const bool zn_size_OK = (zn_size.head < 256.0f);
            const bool test1a = (zn_size.head < normDeltaSubN.head);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2.head >= w2threshold.head) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            }
            else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + dblflt_to_double(DeltaSubNWX) * S); // Xxrd, xr
                DoubleTempZY = (curDoubleIter->y + dblflt_to_double(DeltaSubNWY) * S); // Xxid, xi

                RefIteration = 0;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = double_to_dblflt(S);
                twos = add_dblflt(s, s);

                DeltaSub0DX = double_to_dblflt(DeltaReal / S);
                DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
                DeltaSubNWX = double_to_dblflt(DoubleTempZX / S);
                DeltaSubNWY = double_to_dblflt(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = dblflt_to_double(DeltaSubNWX) * S;
                DoubleTempZY = dblflt_to_double(DeltaSubNWY) * S;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = double_to_dblflt(S);
                twos = add_dblflt(s, s);

                DeltaSub0DX = double_to_dblflt(DeltaReal / S);
                DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
                DeltaSubNWX = double_to_dblflt(DoubleTempZX / S);
                DeltaSubNWY = double_to_dblflt(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        }
        else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = dblflt_to_double(DeltaSubNWX);
            double DeltaSubNWYOrig = dblflt_to_double(DeltaSubNWY);

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * curDoubleIter->x * 2 -
                DeltaSubNWYOrig * curDoubleIter->y * 2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y * 2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x * 2 +
                DeltaImaginary / S;

            ++RefIteration;
            curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const double tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                curDoubleIter->y +
                DoubleTempDeltaSubNWY * S; // Xxid

            const double zn_size =
                tempZX * tempZX + tempZY * tempZY;

            if (zn_size > 256.0) {
                break;
            }

            const double TwoS = S * S;
            const double normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * TwoS +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * TwoS;

            double DeltaSubNWXNew;
            double DeltaSubNWYNew;

            if (zn_size < normDeltaSubN ||
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            s = double_to_dblflt(S);
            twos = add_dblflt(s, s);

            DeltaSub0DX = double_to_dblflt(DeltaReal / S);
            DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
            DeltaSubNWX = double_to_dblflt(DeltaSubNWXNew / S);
            DeltaSubNWY = double_to_dblflt(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    OutputIterMatrix[idx] = iter;
}
