////////////////////////////////////////////////////////////////////////////////////////
// Perturbation results
////////////////////////////////////////////////////////////////////////////////////////

template<typename IterType, typename Type, CalcBad Bad>
struct MattPerturbSingleResults {
    MattReferenceSingleIter<Type, Bad>* __restrict__ iters;
    IterType size;
    cudaError_t err;
    IterType PeriodMaybeZero;
    bool own;
    bool AllocHost;

    MattPerturbSingleResults() = delete;

    template<typename Other>
    MattPerturbSingleResults(
        IterType sz,
        IterType PeriodMaybeZero,
        MattReferenceSingleIter<Other, Bad>* in_iters)
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

        check_size<MattReferenceSingleIter<float>, 8>();
        check_size<MattReferenceSingleIter<double>, 16>();
        check_size<MattReferenceSingleIter<HDRFloat<float>>, 16>();
        check_size<MattReferenceSingleIter<HDRFloat<double>>, 32>();
        check_size<MattReferenceSingleIter<HDRFloat<CudaDblflt<MattDblflt>>>, 24>();
        check_size<MattReferenceSingleIter<HDRFloat<CudaDblflt<dblflt>>>, 24>();
        check_size<MattReferenceSingleIter<dblflt>, 16>();

        MattReferenceSingleIter<Type, Bad>* tempIters;
        AllocHost = false;
        err = cudaMallocManaged(&tempIters, size * sizeof(MattReferenceSingleIter<Type, Bad>));
        if (err != cudaSuccess) {
            AllocHost = true;
            err = cudaMallocHost(&tempIters, size * sizeof(MattReferenceSingleIter<Type, Bad>));
            if (err != cudaSuccess) {
                size = 0;
                return;
            }
        }

        iters = tempIters;
        cudaMemcpy(iters, in_iters, size * sizeof(MattReferenceSingleIter<Type, Bad>), cudaMemcpyDefault);

        //err = cudaMemAdvise(iters,
        //    size * sizeof(MattReferenceSingleIter<Type>),
        //    cudaMemAdviseSetReadMostly,
        //    0);
        //if (err != cudaSuccess) {
        //    size = 0;
        //    return;
        //}
    }

    MattPerturbSingleResults(const MattPerturbSingleResults& other) {
        iters = reinterpret_cast<MattReferenceSingleIter<Type, Bad>*>(other.iters);
        size = other.size;
        PeriodMaybeZero = other.PeriodMaybeZero;
        own = false;
        AllocHost = other.AllocHost;
    }

    // funny semantics, copy doesn't own the pointers.
    template<class Other>
    MattPerturbSingleResults(const MattPerturbSingleResults<IterType, Other, Bad>& other) {
        iters = reinterpret_cast<MattReferenceSingleIter<Type, Bad>*>(other.iters);
        size = other.size;
        PeriodMaybeZero = other.PeriodMaybeZero;
        own = false;
        AllocHost = other.AllocHost;
    }

    uint32_t CheckValid() const {
        return err;
    }

    MattPerturbSingleResults(MattPerturbSingleResults&& other) = delete;
    MattPerturbSingleResults& operator=(const MattPerturbSingleResults& other) = delete;
    MattPerturbSingleResults& operator=(MattPerturbSingleResults&& other) = delete;

    ~MattPerturbSingleResults() {
        if (own) {
            if (iters != nullptr) {
                if (!AllocHost) {
                    cudaFree(iters);
                }
                else {
                    cudaFreeHost(iters);
                }
            }
        }
    }
};

