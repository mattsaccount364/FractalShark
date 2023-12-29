////////////////////////////////////////////////////////////////////////////////////////
// Perturbation results
////////////////////////////////////////////////////////////////////////////////////////

template<typename IterType, typename Type, PerturbExtras PExtras>
class GPUPerturbSingleResults : public TemplateHelpers<IterType, Type, PExtras> {
public:
    template<typename IterType, class Type, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, Type, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    GPUPerturbSingleResults() = delete;

    template<typename Other>
    GPUPerturbSingleResults(
        IterType CompressedOrbitSize,
        IterType UncompressedOrbitSize,
        IterType PeriodMaybeZero,
        Type OrbitXLow,
        Type OrbitYLow,
        const GPUReferenceIter<Other, PExtras>* in_iters)
        : FullOrbit{},
        OrbitSize(CompressedOrbitSize),
        UncompressedItersInOrbit(UncompressedOrbitSize),
        PeriodMaybeZero(PeriodMaybeZero),
        OrbitXLow(OrbitXLow),
        OrbitYLow(OrbitYLow),
        err(cudaSuccess),
        own(true),
        AllocHost(false) {

        check_size<HDRFloat<float>, 8>();
        check_size<HDRFloat<double>, 16>();
        check_size<dblflt, double>();
        check_size<MattDblflt, double>();
        check_size<MattDblflt, dblflt>();
        check_size<CudaDblflt<MattDblflt>, double>();
        check_size<CudaDblflt<MattDblflt>, CudaDblflt<dblflt>>();
        check_size<HDRFloat<CudaDblflt<MattDblflt>>, HDRFloat<CudaDblflt<dblflt>>>();
        check_size<Type, Other>();

        check_size<GPUReferenceIter<float>, 8>();
        check_size<GPUReferenceIter<double>, 16>();
        check_size<GPUReferenceIter<HDRFloat<float>>, 16>();
        check_size<GPUReferenceIter<HDRFloat<double>>, 32>();
        check_size<GPUReferenceIter<HDRFloat<CudaDblflt<MattDblflt>>>, 24>();
        check_size<GPUReferenceIter<HDRFloat<CudaDblflt<dblflt>>>, 24>();
        check_size<GPUReferenceIter<dblflt>, 16>();

        GPUReferenceIter<Type, PExtras>* tempIters;
        AllocHost = false;
        err = cudaMallocManaged(&tempIters, OrbitSize * sizeof(GPUReferenceIter<Type, PExtras>));
        if (err != cudaSuccess) {
            AllocHost = true;
            err = cudaMallocHost(&tempIters, OrbitSize * sizeof(GPUReferenceIter<Type, PExtras>));
            if (err != cudaSuccess) {
                OrbitSize = 0;
                return;
            }
        }

        FullOrbit = tempIters;

        // Cast to void -- it's logically const
        cudaMemcpy((void*)FullOrbit, in_iters, OrbitSize * sizeof(GPUReferenceIter<Type, PExtras>), cudaMemcpyDefault);

        //err = cudaMemAdvise(FullOrbit,
        //    size * sizeof(GPUReferenceIter<Type>),
        //    cudaMemAdviseSetReadMostly,
        //    0);
        //if (err != cudaSuccess) {
        //    size = 0;
        //    return;
        //}
    }

    GPUPerturbSingleResults(const GPUPerturbSingleResults& other)
        :
        FullOrbit{ reinterpret_cast<const GPUReferenceIter<Type, PExtras>*>(other.FullOrbit) },
        OrbitSize{ other.OrbitSize },
        UncompressedItersInOrbit{ other.UncompressedItersInOrbit },
        PeriodMaybeZero{ other.PeriodMaybeZero },
        OrbitXLow{ other.OrbitXLow },
        OrbitYLow{ other.OrbitYLow },
        own{},
        AllocHost{ other.AllocHost } {
    }

    // funny semantics, copy doesn't own the pointers.
    template<class Other>
    GPUPerturbSingleResults(const GPUPerturbSingleResults<IterType, Other, PExtras>& other) {
        FullOrbit = reinterpret_cast<const GPUReferenceIter<Type, PExtras>*>(other.FullOrbit);
        OrbitSize = other.OrbitSize;
        UncompressedItersInOrbit = other.UncompressedItersInOrbit;
        PeriodMaybeZero = other.PeriodMaybeZero;
        OrbitXLow = other.OrbitXLow;
        OrbitYLow = other.OrbitYLow;        
        own = false;
        AllocHost = other.AllocHost;
    }

    uint32_t CheckValid() const {
        return err;
    }

    GPUPerturbSingleResults(GPUPerturbSingleResults&& other) = delete;
    GPUPerturbSingleResults& operator=(const GPUPerturbSingleResults& other) = delete;
    GPUPerturbSingleResults& operator=(GPUPerturbSingleResults&& other) = delete;

    ~GPUPerturbSingleResults() {
        if (own) {
            if (FullOrbit != nullptr) {
                if (!AllocHost) {
                    cudaFree((void*)FullOrbit);
                }
                else {
                    cudaFreeHost((void*)FullOrbit);
                }
            }
        }
    }

    //const GPUReferenceIter<T, PExtras>* GetFullOrbit() const {
    //    return FullOrbit;
    //}

    CUDA_CRAP
    IterType GetCountOrbitEntries() const {
        return UncompressedItersInOrbit;
    }

    CUDA_CRAP
    IterType GetPeriodMaybeZero() const {
        return PeriodMaybeZero;
    }

    CUDA_CRAP
    void GetIter(IterType uncompressed_index, Type &x, Type &y) const {
        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            x = FullOrbit[uncompressed_index].x;
            y = FullOrbit[uncompressed_index].y;
        }
        else {
            auto ret = GetCompressedComplex(uncompressed_index);
            x = ret.getRe();
            y = ret.getIm();
        }
    }

    // Returns the value multiplied by two
    CUDA_CRAP
    void GetIterX2(IterType index, Type &x, Type &y) const {
        GetIter(index, x, y);
        x = x * Type{ 2 };
        y = y * Type{ 2 };
    }

    CUDA_CRAP
    HDRFloatComplex<SubType> GetCompressedComplex(IterType uncompressed_index) const {

        // Do a binary search.  Given the uncompressed index, search the compressed
        // FullOrbit array for the nearest UncompressedIndex that's less than or equal
        // to the provided uncompressed index.  Return the compressed index for that
        // uncompressed index.
        auto BinarySearch = [&](IterType uncompressed_index) {
            IterType low = 0;
            IterType high = OrbitSize - 1;

            while (low <= high) {
                IterType mid = (low + high) / 2;

                if (FullOrbit[mid].CompressionIndex < uncompressed_index) {
                    low = mid + 1;
                }
                else if (FullOrbit[mid].CompressionIndex > uncompressed_index) {
                    high = mid - 1;
                }
                else {
                    return mid;
                }
            }

            return high;
            };


        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        Type zx = FullOrbit[BestCompressedIndexGuess].x;
        Type zy = FullOrbit[BestCompressedIndexGuess].y;

        for (IterType cur_uncompressed_index = FullOrbit[BestCompressedIndexGuess].CompressionIndex;
            cur_uncompressed_index < UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            if (cur_uncompressed_index == uncompressed_index) {
                return { zx, zy };
            }

            auto zx_old = zx;
            zx = zx * zx - zy * zy + OrbitXLow;
            HdrReduce(zx);
            zy = Type{ 2 } *zx_old * zy + OrbitYLow;
            HdrReduce(zy);
        }

        // TODO This is an error
        return {};
    }

private:
    const GPUReferenceIter<Type, PExtras>* __restrict__ FullOrbit;

    IterType OrbitSize;
    IterType UncompressedItersInOrbit;
    IterType PeriodMaybeZero;
    Type OrbitXLow;
    Type OrbitYLow;

    cudaError_t err;
    bool own;
    bool AllocHost;
};

