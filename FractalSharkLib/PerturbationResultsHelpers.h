#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

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

    CompressionHelper(const PerturbationResults<IterType, T, PExtras>& results) :
        results(results),
        CachedIter1{},
        CachedIter2{}
    {
    }

    CompressionHelper(const CompressionHelper& other) = default;

    template<class U>
    HDRFloatComplex<U> GetCompressedComplex(size_t uncompressed_index) const {

        auto runOneIter = [&](auto &zx, auto &zy) {
            auto zx_old = zx;
            zx = zx * zx - zy * zy + results.m_OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2 } *zx_old * zy + results.m_OrbitYLow;
            HdrReduce(zy);
        };

        if (CachedIter1.UncompressedIter == uncompressed_index) {
            return { CachedIter1.zx, CachedIter1.zy };
        }

        if (CachedIter2.UncompressedIter == uncompressed_index) {
            return { CachedIter2.zx, CachedIter2.zy };
        }

        auto LinearScan = [&](CachedIter<T>& iter, CachedIter<T>& other) -> bool {
            if (iter.UncompressedIter + 1 == uncompressed_index) {
                bool condition =
                    (iter.CompressedIter + 1 < results.m_FullOrbit.GetSize() &&
                    (iter.UncompressedIter + 1 <
                        results.m_FullOrbit[iter.CompressedIter + 1].CompressionIndex)) ||
                    iter.CompressedIter + 1 == results.m_FullOrbit.GetSize();

                if (condition) {
                    other = iter;
                    runOneIter(iter.zx, iter.zy);
                    iter.UncompressedIter = uncompressed_index;
                }
                else {
                    other = iter;
                    iter.CompressedIter++;
                    iter.zx = results.m_FullOrbit[iter.CompressedIter].x;
                    iter.zy = results.m_FullOrbit[iter.CompressedIter].y;
                    iter.UncompressedIter = results.m_FullOrbit[iter.CompressedIter].CompressionIndex;
                }

                return true;
            }

            return false;
        };

        bool ret = LinearScan(CachedIter1, CachedIter2);
        if (ret) {
            return { CachedIter1.zx, CachedIter1.zy };
        }

        ret = LinearScan(CachedIter2, CachedIter1);
        if (ret) {
            return { CachedIter2.zx, CachedIter2.zy };
        }

        // Do a binary search.  Given the uncompressed index, search the compressed
        // m_FullOrbit array for the nearest UncompressedIndex that's less than or equal
        // to the provided uncompressed index.  Return the compressed index for that
        // uncompressed index.
        auto BinarySearch = [&](size_t uncompressed_index) {
            size_t low = 0;
            size_t high = results.m_FullOrbit.GetSize() - 1;

            while (low <= high) {
                size_t mid = (low + high) / 2;

                if (results.m_FullOrbit[mid].CompressionIndex < uncompressed_index) {
                    low = mid + 1;
                }
                else if (results.m_FullOrbit[mid].CompressionIndex > uncompressed_index) {
                    high = mid - 1;
                }
                else {
                    return mid;
                }
            }

            return high;
        };


        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        T zx = results.m_FullOrbit[BestCompressedIndexGuess].x;
        T zy = results.m_FullOrbit[BestCompressedIndexGuess].y;

        for (size_t cur_uncompressed_index = results.m_FullOrbit[BestCompressedIndexGuess].CompressionIndex;
            cur_uncompressed_index < results.m_UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            if (cur_uncompressed_index == uncompressed_index) {
                CachedIter2 = CachedIter1;
                CachedIter1 = CachedIter<T>(zx, zy, cur_uncompressed_index, BestCompressedIndexGuess);
                return { zx, zy };
            }

            runOneIter(zx, zy);
        }

        return {};
    }

private:
    template<typename T>
    struct CachedIter {
        CachedIter() :
            zx{},
            zy{},
            UncompressedIter{ UINT64_MAX - 1 },
            CompressedIter{ UINT64_MAX - 1 } {
        }

        CachedIter(T zx, T zy, size_t uncompressed_iter, size_t compressed_iter) :
            zx(zx),
            zy(zy),
            UncompressedIter(uncompressed_iter),
            CompressedIter(compressed_iter) {
        }

        CachedIter &operator=(const CachedIter &other) = default;
        CachedIter(const CachedIter &other) = default;

        T zx;
        T zy;
        IterTypeFull UncompressedIter;
        IterTypeFull CompressedIter;
    };

    const PerturbationResults<IterType, T, PExtras>& results;
    mutable CachedIter<T> CachedIter1;
    mutable CachedIter<T> CachedIter2;
};

template<typename IterType, class T, PerturbExtras PExtras>
class IntermediateCompressionHelper : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    IntermediateCompressionHelper(const PerturbationResults<IterType, T, PExtras>& results) :
        results(results),
        CachedIter1{},
        CachedIter2{},
        cx{results.GetHiX()},
        cy{results.GetHiY()},
        Two{2}
    {
        cx.precisionInBits(AuthoritativeReuseExtraPrecisionInBits);
        cy.precisionInBits(AuthoritativeReuseExtraPrecisionInBits);
    }

    IntermediateCompressionHelper(const IntermediateCompressionHelper& other) = default;

    void GetReuseEntries(
        size_t uncompressed_index,
        const HighPrecisionT<HPDestructor::False>*& outX,
        const HighPrecisionT<HPDestructor::False>*& outY) const {

        auto runOneIter = [&](auto& zx, auto& zy) {
            auto zx_old = zx;
            zx = zx * zx - zy * zy + cx;
            zy = Two * zx_old * zy + cy;
        };

        if (CachedIter1.UncompressedIter == uncompressed_index) {
            outX = &CachedIter1.zx;
            outY = &CachedIter1.zy;
        }

        if (CachedIter2.UncompressedIter == uncompressed_index) {
            outX = &CachedIter2.zx;
            outY = &CachedIter2.zy;
        }

        auto LinearScan = [&](auto& iter, auto& other) -> bool {
            if (iter.UncompressedIter + 1 == uncompressed_index) {
                bool condition =
                    (iter.CompressedIter + 1 < results.m_FullOrbit.GetSize() &&
                        (iter.UncompressedIter + 1 <
                            results.m_ReuseIndices[iter.CompressedIter + 1])) ||
                    iter.CompressedIter + 1 == results.m_FullOrbit.GetSize();

                if (condition) {
                    other = iter;
                    runOneIter(iter.zx, iter.zy);
                    iter.UncompressedIter = uncompressed_index;
                }
                else {
                    other = iter;
                    iter.CompressedIter++;
                    iter.zx = results.m_ReuseX[iter.CompressedIter];
                    iter.zy = results.m_ReuseY[iter.CompressedIter];
                    iter.UncompressedIter = results.m_ReuseIndices[iter.CompressedIter];
                }

                return true;
            }

            return false;
        };

        bool ret = LinearScan(CachedIter1, CachedIter2);
        if (ret) {
            outX = &CachedIter1.zx;
            outY = &CachedIter1.zy;
        }

        ret = LinearScan(CachedIter2, CachedIter1);
        if (ret) {
            outX = &CachedIter2.zx;
            outY = &CachedIter2.zy;
        }

        // Do a binary search.  Given the uncompressed index, search the compressed
        // m_FullOrbit array for the nearest UncompressedIndex that's less than or equal
        // to the provided uncompressed index.  Return the compressed index for that
        // uncompressed index.
        auto BinarySearch = [&](size_t uncompressed_index) {
            size_t low = 0;
            size_t high = results.m_ReuseIndices.size() - 1;

            while (low <= high) {
                size_t mid = (low + high) / 2;

                if (results.m_ReuseIndices[mid] < uncompressed_index) {
                    low = mid + 1;
                }
                else if (results.m_ReuseIndices[mid] > uncompressed_index) {
                    high = mid - 1;
                }
                else {
                    return mid;
                }
            }

            return high;
        };

        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        const auto &zx = results.m_ReuseX[BestCompressedIndexGuess];
        const auto &zy = results.m_ReuseY[BestCompressedIndexGuess];

        for (size_t cur_uncompressed_index = results.m_ReuseIndices[BestCompressedIndexGuess];
            cur_uncompressed_index < results.m_UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            if (cur_uncompressed_index == uncompressed_index) {
                CachedIter2 = CachedIter1;
                CachedIter1 = CachedIter<T>(&zx, &zy, cur_uncompressed_index, BestCompressedIndexGuess);
                outX = &CachedIter1.zx;
                outY = &CachedIter1.zy;
            }

            runOneIter(zx, zy);
        }

        return {};
    }

private:
    template<typename T>
    struct CachedIter {
        CachedIter() :
            zx{},
            zy{},
            UncompressedIter{ UINT64_MAX - 1 },
            CompressedIter{ UINT64_MAX - 1 } {
        }

        CachedIter(
            const HighPrecisionT<HPDestructor::False>* zx,
            const HighPrecisionT<HPDestructor::False>* zy,
            size_t uncompressed_iter,
            size_t compressed_iter) :
            zx(zx),
            zy(zy),
            UncompressedIter(uncompressed_iter),
            CompressedIter(compressed_iter) {
        }

        CachedIter& operator=(const CachedIter& other) = default;
        CachedIter(const CachedIter& other) = default;

        // TODO leaks?
        HighPrecisionT<HPDestructor::False> zx;
        HighPrecisionT<HPDestructor::False> zy;
        IterTypeFull UncompressedIter;
        IterTypeFull CompressedIter;
    };

    const PerturbationResults<IterType, T, PExtras>& results;
    mutable CachedIter<T> CachedIter1;
    mutable CachedIter<T> CachedIter2;

    // TODO leaks?
    HighPrecisionT<HPDestructor::False> cx;
    HighPrecisionT<HPDestructor::False> cy;
    HighPrecisionT<HPDestructor::False> Two;
};