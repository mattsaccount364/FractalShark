
#if 0
// TODO Do we need this kind of conversion for the 2x32 LAv2 case:
template<typename IterType>
__global__
void mandel_2x_float_perturb_setup(GPUPerturbSingleResults<IterType, dblflt> PerturbDblFlt)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0)
        return;

    for (IterType i = 0; i < PerturbDblFlt.size; i++) {
        PerturbDblFlt.iters[i].x = add_float_to_dblflt(PerturbDblFlt.iters[i].x.y, PerturbDblFlt.iters[i].x.x);
        PerturbDblFlt.iters[i].y = add_float_to_dblflt(PerturbDblFlt.iters[i].y.y, PerturbDblFlt.iters[i].y.x);
    }
}

template<typename IterType>
__global__
void mandel_2x_float_perturb(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    GPUPerturbSingleResults<IterType, dblflt> PerturbDblFlt,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    dblflt centerX,
    dblflt centerY,
    IterType n_iterations)
{

    //int X = blockIdx.x * blockDim.x + threadIdx.x;
    //int Y = blockIdx.y * blockDim.y + threadIdx.y;

    //if (X >= width || Y >= height)
    //    return;

    ////size_t idx = width * (height - Y - 1) + X;
    //size_t idx = width * Y + X;
    //size_t idx = ConvertLocToIndex(X, Y, width);

    //if (OutputIterMatrix[idx] != 0) {
    //    return;
    //}

    //IterType iter = 0;
    //IterType RefIteration = 0;
    //double DeltaReal = dx * X - centerX;
    //double DeltaImaginary = -dy * Y - centerY;

    //double DeltaSub0X = DeltaReal;
    //double DeltaSub0Y = DeltaImaginary;
    //double DeltaSubNX = 0;
    //double DeltaSubNY = 0;

    //while (iter < n_iterations) {
    //    const double DeltaSubNXOrig = DeltaSubNX;
    //    const double DeltaSubNYOrig = DeltaSubNY;

    //    DeltaSubNX =
    //        DeltaSubNXOrig * (results_x2[RefIteration] + DeltaSubNXOrig) -
    //        DeltaSubNYOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
    //        DeltaSub0X;
    //    DeltaSubNY =
    //        DeltaSubNXOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
    //        DeltaSubNYOrig * (results_x2[RefIteration] + DeltaSubNXOrig) +
    //        DeltaSub0Y;

    //    ++RefIteration;

    //    const double tempZX = results_x[RefIteration] + DeltaSubNX;
    //    const double tempZY = results_y[RefIteration] + DeltaSubNY;
    //    const double zn_size = tempZX * tempZX + tempZY * tempZY;
    //    const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

    //    if (zn_size > 256) {
    //        break;
    //    }

    //    if (zn_size < normDeltaSubN ||
    //        RefIteration == sz - 1) {
    //        DeltaSubNX = tempZX;
    //        DeltaSubNY = tempZY;
    //        RefIteration = 0;
    //    }

    //    ++iter;
    //}

    //OutputIterMatrix[idx] = iter;

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

    IterType iter = 0;
    IterType RefIteration = 0;

    dblflt Two = add_float_to_dblflt(0, 2);

    dblflt X2 = add_float_to_dblflt(X, 0);
    dblflt Y2 = add_float_to_dblflt(Y, 0);
    dblflt MinusY2 = add_float_to_dblflt(-Y, 0);

    dblflt DeltaReal = sub_dblflt(mul_dblflt(dx, X2), centerX);
    dblflt DeltaImaginary = sub_dblflt(mul_dblflt(dy, MinusY2), centerY);

    dblflt DeltaSub0X = DeltaReal;
    dblflt DeltaSub0Y = DeltaImaginary;
    dblflt DeltaSubNX, DeltaSubNY;

    IterType MaxRefIteration = PerturbDblFlt.size - 1;

    DeltaSubNX = add_float_to_dblflt(0, 0);
    DeltaSubNY = add_float_to_dblflt(0, 0);

    while (iter < n_iterations) {
        GPUReferenceIter<dblflt>* CurIter = &PerturbDblFlt.iters[RefIteration];

        const dblflt DeltaSubNXOrig = DeltaSubNX;
        const dblflt DeltaSubNYOrig = DeltaSubNY;

        const dblflt tempX = mul_dblflt(CurIter->x, Two);
        const dblflt tempY = mul_dblflt(CurIter->y, Two);

        const dblflt tempTermX1 = add_dblflt(tempX, DeltaSubNXOrig);
        const dblflt tempTermX2 = add_dblflt(tempY, DeltaSubNYOrig);

        DeltaSubNX =
            sub_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX1),
                mul_dblflt(DeltaSubNYOrig, tempTermX2)
            );
        DeltaSubNX = add_dblflt(DeltaSubNX, DeltaSub0X);

        DeltaSubNY =
            add_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX2),
                mul_dblflt(DeltaSubNYOrig, tempTermX1)
            );
        DeltaSubNY = add_dblflt(DeltaSubNY, DeltaSub0Y);

        ++RefIteration;
        CurIter = &PerturbDblFlt.iters[RefIteration];

        const dblflt tempZX = add_dblflt(CurIter->x, DeltaSubNX);
        const dblflt tempZY = add_dblflt(CurIter->y, DeltaSubNY);
        const dblflt zn_size = add_dblflt(sqr_dblflt(tempZX), sqr_dblflt(tempZY));
        const dblflt normDeltaSubN = add_dblflt(sqr_dblflt(DeltaSubNX), sqr_dblflt(DeltaSubNY));

        if (zn_size.y > 256) {
            break;
        }

        if (zn_size.y < normDeltaSubN.y ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    OutputIterMatrix[idx] = iter;
}
#endif