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

        check_size<GPUReferenceIter<float, PerturbExtras::Disable>, 8>();
        check_size<GPUReferenceIter<double, PerturbExtras::Disable>, 16>();
        check_size<GPUReferenceIter<HDRFloat<float>, PerturbExtras::Disable>, 16>();
        check_size<GPUReferenceIter<HDRFloat<double>, PerturbExtras::Disable>, 32>();
        check_size<GPUReferenceIter<HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>, 24>();
        check_size<GPUReferenceIter<HDRFloat<CudaDblflt<dblflt>>, PerturbExtras::Disable>, 24>();
        check_size<GPUReferenceIter<dblflt, PerturbExtras::Disable>, 16>();

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
    void GetIterRandom(IterType uncompressed_index, Type &x, Type &y) const {
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

    struct SeqWorkspace {
        CUDA_CRAP
        SeqWorkspace(GPUPerturbSingleResults &results, IterType startUncompressedIndex) {

            if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
                uncompressed_index = startUncompressedIndex;
                compressed_index = startUncompressedIndex + 1;
                zx = results.FullOrbit[startUncompressedIndex].x;
                zy = results.FullOrbit[startUncompressedIndex].y;
                return;
            }
            else {
                compressed_index = results.BinarySearch(startUncompressedIndex);
                uncompressed_index = results.FullOrbit[compressed_index].CompressionIndex;
                zx = results.FullOrbit[compressed_index].x;
                zy = results.FullOrbit[compressed_index].y;

                compressed_index++;

                for (; uncompressed_index < results.UncompressedItersInOrbit;
                    uncompressed_index++) {

                    if (uncompressed_index == startUncompressedIndex) {
                        return;
                    }

                    auto zx_old = zx;
                    zx = zx * zx - zy * zy + results.OrbitXLow;
                    HdrReduce(zx);
                    zy = Type{ 2 } *zx_old * zy + results.OrbitYLow;
                    HdrReduce(zy);
                }

                *this = {};
            }
        }

        CUDA_CRAP SeqWorkspace() : compressed_index{}, uncompressed_index{}, zx{}, zy{} {}

        IterType compressed_index;
        IterType uncompressed_index;
        Type zx;
        Type zy;
    };

    // Returns the value multiplied by two
    CUDA_CRAP
    void GetIterRandomX2(IterType index, Type &x, Type &y) const {
        GetIterRandom(index, x, y);
        x = x * Type{ 2 };
        y = y * Type{ 2 };
    }

    CUDA_CRAP
    void GetIterSeq(SeqWorkspace &workspace) const {
        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            workspace.uncompressed_index++;
            workspace.zx = FullOrbit[workspace.uncompressed_index].x;
            workspace.zy = FullOrbit[workspace.uncompressed_index].y;
        } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            GetCompressedComplexSeq(workspace);           
        }
    }

    CUDA_CRAP
    const GPUReferenceIter<Type, PExtras> *ScaledOnlyGetIter(IterType index) const {
        return &FullOrbit[index];
    }

private:
    // Do a binary search.  Given the uncompressed index, search the compressed
    // FullOrbit array for the nearest UncompressedIndex that's less than or equal
    // to the provided uncompressed index.  Return the compressed index for that
    // uncompressed index.
    CUDA_CRAP
    IterType BinarySearch(IterType uncompressed_index) const {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return uncompressed_index;
        }
        else {
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
        }
    }

    CUDA_CRAP
    HDRFloatComplex<SubType> GetCompressedComplex(IterType uncompressed_index) const {
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

        // TODO This is an error condition, but we don't have a good way to report it
        return {};
    }

    CUDA_CRAP
    void GetCompressedComplexSeq(SeqWorkspace &workspace) const {
        workspace.uncompressed_index++;
        // compressed already points to the next one

        TODO do we ever walk off the end here
                    //some notes
            //    it looks GPU specific 
            //    64-bit specific
            //    non-deterministic
            //    maybe copying compressed orbit to gpu wrong / uninitialized memory somewhere?


        auto nextIndex = FullOrbit[workspace.compressed_index].CompressionIndex;
        if (nextIndex == workspace.uncompressed_index) {
            workspace.zx = FullOrbit[workspace.compressed_index].x;
            workspace.zy = FullOrbit[workspace.compressed_index].y;
            workspace.compressed_index++;
        }
        else {
            auto zx_old = workspace.zx;
            workspace.zx = workspace.zx * workspace.zx - workspace.zy * workspace.zy + OrbitXLow;
            HdrReduce(workspace.zx);
            workspace.zy = Type{ 2 } *zx_old * workspace.zy + OrbitYLow;
            HdrReduce(workspace.zy);
        }
    }

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

