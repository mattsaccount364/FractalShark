////////////////////////////////////////////////////////////////////////////////////////
// Perturbation results
////////////////////////////////////////////////////////////////////////////////////////

template<typename IterType, typename Type, PerturbExtras PExtras>
struct GPUPerturbSingleResults {
    const GPUReferenceIter<Type, PExtras>* __restrict__ iters;
    IterType size;
    cudaError_t err;
    IterType PeriodMaybeZero;
    bool own;
    bool AllocHost;

    GPUPerturbSingleResults() = delete;

    template<typename Other>
    GPUPerturbSingleResults(
        IterType sz,
        IterType PeriodMaybeZero,
        const GPUReferenceIter<Other, PExtras>* in_iters)
        : iters(nullptr),
        size(sz),
        err(cudaSuccess),
        PeriodMaybeZero(PeriodMaybeZero),
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
        err = cudaMallocManaged(&tempIters, size * sizeof(GPUReferenceIter<Type, PExtras>));
        if (err != cudaSuccess) {
            AllocHost = true;
            err = cudaMallocHost(&tempIters, size * sizeof(GPUReferenceIter<Type, PExtras>));
            if (err != cudaSuccess) {
                size = 0;
                return;
            }
        }

        iters = tempIters;

        // Cast to void -- it's logically const
        cudaMemcpy((void*)iters, in_iters, size * sizeof(GPUReferenceIter<Type, PExtras>), cudaMemcpyDefault);

        //err = cudaMemAdvise(iters,
        //    size * sizeof(GPUReferenceIter<Type>),
        //    cudaMemAdviseSetReadMostly,
        //    0);
        //if (err != cudaSuccess) {
        //    size = 0;
        //    return;
        //}
    }

    GPUPerturbSingleResults(const GPUPerturbSingleResults& other) {
        iters = reinterpret_cast<const GPUReferenceIter<Type, PExtras>*>(other.iters);
        size = other.size;
        PeriodMaybeZero = other.PeriodMaybeZero;
        own = false;
        AllocHost = other.AllocHost;
    }

    // funny semantics, copy doesn't own the pointers.
    template<class Other>
    GPUPerturbSingleResults(const GPUPerturbSingleResults<IterType, Other, PExtras>& other) {
        iters = reinterpret_cast<const GPUReferenceIter<Type, PExtras>*>(other.iters);
        size = other.size;
        PeriodMaybeZero = other.PeriodMaybeZero;
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
            if (iters != nullptr) {
                if (!AllocHost) {
                    cudaFree((void*)iters);
                }
                else {
                    cudaFreeHost((void*)iters);
                }
            }
        }
    }
};

