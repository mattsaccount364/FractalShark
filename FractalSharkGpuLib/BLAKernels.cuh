
template<class HDRFloatType>
struct SharedMemStruct {
    using GPUBLA_TYPE = BLA<HDRFloatType>;
    const GPUBLA_TYPE* __restrict__ altB[32];
    //GPUBLA_TYPE nullBla;
    //struct {
    //    //HDRFloatType curBR2[16];
    //    //GPUReferenceIter<HDRFloatType> CurResult;
    //    //HDRFloatType NextX1;
    //    //HDRFloatType NextY1;
    //} PerThread[NB_THREADS_W][NB_THREADS_H];
};

template<typename IterType, int32_t LM2>
__global__
void mandel_1x_double_perturb_bla(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    GPUPerturbSingleResults<IterType, double> PerturbDouble,
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

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
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
    double DeltaReal = dx * X - centerX;
    double DeltaImaginary = -dy * Y - centerY;

    double DeltaSub0X = DeltaReal;
    double DeltaSub0Y = DeltaImaginary;
    double DeltaSubNX = 0;
    double DeltaSubNY = 0;
    double DeltaNormSquared = 0;

    while (iter < n_iterations) {
        const BLA<double>* b = nullptr;
        while ((b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared)) != nullptr) {
            int l = b->getL();

            // TODO this first RefIteration + l check bugs me
            if (RefIteration + l >= PerturbDouble.GetCountOrbitEntries()) {
                break;
            }

            if (iter + l >= n_iterations) {
                break;
            }

            iter += l;
            RefIteration += l;

            b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

            double tempZX;
            double tempZY;
            PerturbDouble.GetIter(RefIteration, tempZX, tempZY);
            tempZX = tempZX + DeltaSubNX;
            tempZY = tempZY + DeltaSubNY;

            const double normSquared = tempZX * tempZX + tempZY * tempZY;
            DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

            if (normSquared > 256) {
                break;
            }

            if (normSquared < DeltaNormSquared ||
                RefIteration >= PerturbDouble.GetCountOrbitEntries() - 1) {
                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;
                DeltaNormSquared = normSquared;
                RefIteration = 0;
            }
        }

        if (iter >= n_iterations) {
            break;
        }

        const double DeltaSubNXOrig = DeltaSubNX;
        const double DeltaSubNYOrig = DeltaSubNY;

        double tempZX;
        double tempZY;
        PerturbDouble.GetIter(RefIteration, tempZX, tempZY);

        DeltaSubNX = DeltaSubNXOrig * (tempZX * 2 + DeltaSubNXOrig) -
            DeltaSubNYOrig * (tempZY * 2 + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY = DeltaSubNXOrig * (tempZY * 2 + DeltaSubNYOrig) +
            DeltaSubNYOrig * (tempZX * 2 + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;

        PerturbDouble.GetIter(RefIteration, tempZX, tempZY);
        tempZX = tempZX + DeltaSubNX;
        tempZY = tempZY + DeltaSubNY;

        const double normSquared = tempZX * tempZX + tempZY * tempZY;
        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (normSquared > 256) {
            break;
        }

        if (normSquared < DeltaNormSquared ||
            RefIteration >= PerturbDouble.GetCountOrbitEntries() - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            DeltaNormSquared = normSquared;
            RefIteration = 0;
        }

        ++iter;
    }

    OutputIterMatrix[idx] = iter;
}

//#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

//#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
//#define PXL_GLOBAL_PTR   "l"
//#else
//#define PXL_GLOBAL_PTR   "r"
//#endif

//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l1(const void* const ptr)
//{
//    asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}
//
//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
//{
//    asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}
//
//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l2(const void* const ptr)
//{
//    asm("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}

template<typename IterType, class HDRFloatType, int32_t LM2>
__global__
void
//__launch_bounds__(NB_THREADS_W * NB_THREADS_H, 2)
mandel_1xHDR_float_perturb_bla(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    GPUPerturbSingleResults<IterType, HDRFloatType> Perturb,
    GPU_BLAS<IterType, HDRFloatType, BLA<HDRFloatType>, LM2> blas,
    int width,
    int height,
    const HDRFloatType cx,
    const HDRFloatType cy,
    const HDRFloatType dx,
    const HDRFloatType dy,
    const HDRFloatType centerX,
    const HDRFloatType centerY,
    IterType n_iterations)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    //size_t idx = width * Y + X;
    size_t idx = ConvertLocToIndex(X, Y, width);

    //if (OutputIterMatrix[idx] != 0) { 
    //    return;
    //}

    using GPUBLA_TYPE = BLA<HDRFloatType>;
    char __shared__ SharedMem[sizeof(SharedMemStruct<HDRFloatType>)];
    auto* shared =
        reinterpret_cast<SharedMemStruct<HDRFloatType>*>(SharedMem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        GPUBLA_TYPE** elts = blas.GetB();

        for (size_t i = 0; i < blas.m_NumLevels; i++) {
            shared->altB[i] = elts[i];
        }

        //const GPUBLA_TYPE nullBla{
        //    HDRFloatType(),
        //    HDRFloatType(1.0f),
        //    HDRFloatType(),
        //    HDRFloatType(),
        //    HDRFloatType(),
        //    0 };
        //shared->nullBla = nullBla;
    }

    __syncthreads();

    IterType iter = 0;
    IterType RefIteration = 0;
    const HDRFloatType DeltaReal = dx * X - centerX;
    const HDRFloatType DeltaImaginary = -dy * Y - centerY;

    const HDRFloatType DeltaSub0X = DeltaReal;
    const HDRFloatType DeltaSub0Y = DeltaImaginary;
    HDRFloatType DeltaSubNX = HDRFloatType(0);
    HDRFloatType DeltaSubNY = HDRFloatType(0);
    HDRFloatType DeltaNormSquared = HDRFloatType(0);
    const HDRFloatType Two = HDRFloatType(2);

    while (iter < n_iterations) {
        //auto* next1X = &shared->PerThread[threadIdx.x][threadIdx.y].NextX1;
        //auto* next1Y = &shared->PerThread[threadIdx.x][threadIdx.y].NextY1;
        //__pipeline_memcpy_async(
        //    next1X,
        //    &Perturb.iters[RefIteration + 1].x,
        //    sizeof(Perturb.iters[RefIteration + 1].x),
        //    0);
        //__pipeline_memcpy_async(
        //    next1Y,
        //    &Perturb.iters[RefIteration + 1].y,
        //    sizeof(Perturb.iters[RefIteration + 1].y),
        //    0);
        //__pipeline_commit();

        const HDRFloatType DeltaSubNXOrig = DeltaSubNX;
        const HDRFloatType DeltaSubNYOrig = DeltaSubNY;

        //__prefetch_global_l2(&Perturb.iters[RefIteration + 1].x);
        //__prefetch_global_l2(&Perturb.iters[RefIteration + 1].y);

        HDRFloatType tempMulX2;
        HDRFloatType tempMulY2;
        Perturb.GetIterX2(RefIteration, tempMulX2, tempMulY2);

        ++RefIteration;

        //if (RefIteration >= Perturb.GetCountOrbitEntries()) {
        //    // TODO this first RefIteration + l check bugs me
        //    iter = 255;
        //    break;
        //}

        const auto tempSum1 = (tempMulY2 + DeltaSubNYOrig);
        const auto tempSum2 = (tempMulX2 + DeltaSubNXOrig);

        //DeltaSubNX = DeltaSubNXOrig * tempSum2 -
        //    DeltaSubNYOrig * tempSum1 +
        //    DeltaSub0X;
        //HdrReduce(DeltaSubNX);

        //DeltaSubNX = HDRFloatType::custom_perturb1<false>(
        //    DeltaSubNXOrig,
        //    tempSum2,
        //    DeltaSubNYOrig,
        //    tempSum1,
        //    DeltaSub0X);

        //DeltaSubNY = HDRFloatType::custom_perturb1<true>(
        //    DeltaSubNXOrig,
        //    tempSum1,
        //    DeltaSubNYOrig,
        //    tempSum2,
        //    DeltaSub0Y);

        HDRFloatType::custom_perturb2(
            DeltaSubNX,
            DeltaSubNY,
            DeltaSubNXOrig,
            tempSum2,
            DeltaSubNYOrig,
            tempSum1,
            DeltaSub0X,
            DeltaSub0Y);

        //DeltaSubNY = DeltaSubNXOrig * tempSum1 +
        //    DeltaSubNYOrig * tempSum2 +
        //    DeltaSub0Y;
        //HdrReduce(DeltaSubNY);

        //__pipeline_wait_prior(0);

        HDRFloatType tempVal1X;
        HDRFloatType tempVal1Y;
        Perturb.GetIter(RefIteration, tempVal1X, tempVal1Y);

        const HDRFloatType tempZX = tempVal1X + DeltaSubNX;
        const HDRFloatType tempZY = tempVal1Y + DeltaSubNY;
        HDRFloatType normSquared = tempZX * tempZX + tempZY * tempZY;
        HdrReduce(normSquared);

        if (HdrCompareToBothPositiveReducedLT<HDRFloatType, 256>(normSquared) && iter < n_iterations) {
            DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
            HdrReduce(DeltaNormSquared);

            if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                RefIteration >= Perturb.GetCountOrbitEntries() - 1) {
                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;
                DeltaNormSquared = normSquared;
                RefIteration = 0;
            }

            ++iter;
        }
        else {
            break;
        }

        const BLA<HDRFloatType>* b = nullptr;

        for (;;) {
            b = blas.LookupBackwards(
                shared->altB,
                /*shared->PerThread[threadIdx.x][threadIdx.y].curBR2,*/
                //&shared->nullBla,
                RefIteration,
                DeltaNormSquared);
            if (b == nullptr) {
                break;
            }

            const int l = b->getL();

            // TODO this first RefIteration + l check bugs me
            const bool res1 = (RefIteration + l >= Perturb.GetCountOrbitEntries());
            const bool res2 = (iter + l >= n_iterations);
            const bool res3 = (RefIteration + l < Perturb.GetCountOrbitEntries() - 1);
            //const bool res4 = l == 0; // nullBla
            const bool res12 = (/*res4 || */res1 || res2) == false;
            if (res12 && res3) {
                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                HdrReduce(DeltaNormSquared);
                continue;
            }
            else if (res12 && !res3) {
                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                //__pipeline_memcpy_async(
                //    &shared->PerThread[threadIdx.x][threadIdx.y].CurResult.x,
                //    &Perturb.iters[RefIteration].x,
                //    sizeof(Perturb.iters[RefIteration].x),
                //    0);
                //__pipeline_memcpy_async(
                //    &shared->PerThread[threadIdx.x][threadIdx.y].CurResult.y,
                //    &Perturb.iters[RefIteration].y,
                //    sizeof(Perturb.iters[RefIteration].y),
                //    0);
                //__pipeline_commit();
                //__pipeline_wait_prior(0);

                //HDRFloatType tempZX = shared->PerThread[threadIdx.x][threadIdx.y].CurResult.x + DeltaSubNX;
                //HDRFloatType tempZY = shared->PerThread[threadIdx.x][threadIdx.y].CurResult.y + DeltaSubNY;
                HDRFloatType tempZX;
                HDRFloatType tempZY;
                Perturb.GetIter(RefIteration, tempZX, tempZY);

                DeltaSubNX = tempZX + DeltaSubNX;
                DeltaSubNY = tempZY + DeltaSubNY;

                DeltaNormSquared = tempZX.square_mutable() + tempZY.square_mutable();
                HdrReduce(DeltaNormSquared);
                RefIteration = 0;
                break;
            }
            else {
                break;
            }
        }
    }

    OutputIterMatrix[idx] = iter;
}
