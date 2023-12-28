#pragma once

template<typename IterType, class T, PerturbExtras PExtras>
class TemplateHelpers {

protected:
    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<
        std::is_fundamental<T>::value,
        T>::type;

    static constexpr bool IsHDR =
        std::is_same<T, ::HDRFloat<float>>::value ||
        std::is_same<T, ::HDRFloat<double>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<dblflt>>>::value;

    template<class LocalSubType>
    using HDRFloatComplex =
        std::conditional<IsHDR,
        ::HDRFloatComplex<LocalSubType>,
        ::FloatComplex<LocalSubType>>::type;
};

template<typename IterType, class T, PerturbExtras PExtras>
class GPUPerturbResults;

template<typename IterType, class T, PerturbExtras PExtras>
class CompressionHelper : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    const PerturbationResults<IterType, T, PExtras>& results;

    CompressionHelper(const PerturbationResults<IterType, T, PExtras>& results) :
        results(results) {
    }

    CompressionHelper(const CompressionHelper& other) = default;

    template<class U>
    HDRFloatComplex<U> GetCompressedComplex(size_t uncompressed_index) const {

        // Do a binary search.  Given the uncompressed index, search the compressed
        // FullOrbit array for the nearest UncompressedIndex that's less than or equal
        // to the provided uncompressed index.  Return the compressed index for that
        // uncompressed index.
        auto BinarySearch = [&](size_t uncompressed_index) {
            size_t low = 0;
            size_t high = results.FullOrbit.size() - 1;

            while (low <= high) {
                size_t mid = (low + high) / 2;

                if (results.FullOrbit[mid].CompressionIndex < uncompressed_index) {
                    low = mid + 1;
                }
                else if (results.FullOrbit[mid].CompressionIndex > uncompressed_index) {
                    high = mid - 1;
                }
                else {
                    return mid;
                }
            }

            return high;
            };


        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        T zx = results.FullOrbit[BestCompressedIndexGuess].x;
        T zy = results.FullOrbit[BestCompressedIndexGuess].y;

        for (size_t cur_uncompressed_index = results.FullOrbit[BestCompressedIndexGuess].CompressionIndex;
            cur_uncompressed_index < results.UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            if (cur_uncompressed_index == uncompressed_index) {
                return { zx, zy };
            }

            auto zx_old = zx;
            zx = zx * zx - zy * zy + results.OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2 } *zx_old * zy + results.OrbitYLow;
            HdrReduce(zy);
        }

        return {};
    }
};

template<typename IterType, class T, PerturbExtras PExtras>
class GPUPerturbResults;

template<typename IterType, class T, PerturbExtras PExtras>
class GPUCompressionHelper : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    const GPUPerturbResults<IterType, T, PExtras>& results;

    CUDA_CRAP GPUCompressionHelper(const GPUPerturbResults<IterType, T, PExtras>& results) :
        results(results) {
    }

    CUDA_CRAP GPUCompressionHelper(const GPUCompressionHelper& other) = default;

    template<class U>
    CUDA_CRAP HDRFloatComplex<U> GetCompressedComplex(size_t uncompressed_index) const {

        // Do a binary search.  Given the uncompressed index, search the compressed
        // FullOrbit array for the nearest UncompressedIndex that's less than or equal
        // to the provided uncompressed index.  Return the compressed index for that
        // uncompressed index.
        auto BinarySearch = [&](size_t uncompressed_index) {
            size_t low = 0;
            size_t high = results.FullOrbit.size() - 1;

            while (low <= high) {
                size_t mid = (low + high) / 2;

                if (results.FullOrbit[mid].CompressionIndex < uncompressed_index) {
                    low = mid + 1;
                }
                else if (results.FullOrbit[mid].CompressionIndex > uncompressed_index) {
                    high = mid - 1;
                }
                else {
                    return mid;
                }
            }

            return high;
            };


        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        T zx = results.FullOrbit[BestCompressedIndexGuess].x;
        T zy = results.FullOrbit[BestCompressedIndexGuess].y;

        for (size_t cur_uncompressed_index = results.FullOrbit[BestCompressedIndexGuess].CompressionIndex;
            cur_uncompressed_index < results.UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            assert(BestCompressedIndexGuess < results.FullOrbit.size());
            if (BestCompressedIndexGuess < results.FullOrbit.size() - 1) {
                assert(cur_uncompressed_index < results.FullOrbit[BestCompressedIndexGuess + 1].CompressionIndex);
            }

            if (cur_uncompressed_index == uncompressed_index) {
                return { zx, zy };
            }

            auto zx_old = zx;
            zx = zx * zx - zy * zy + results.OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2 } *zx_old * zy + results.OrbitYLow;
            HdrReduce(zy);
        }

        return {};
    }
};