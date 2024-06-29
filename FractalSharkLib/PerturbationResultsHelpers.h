#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

template<typename IterType, class T, PerturbExtras PExtras>
class TemplateHelpers {

public:
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
class RuntimeDecompressor : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    RuntimeDecompressor(const PerturbationResults<IterType, T, PExtras> &results) :
        results(results),
        CachedIter1{},
        CachedIter2{} {
    }

    RuntimeDecompressor(const RuntimeDecompressor &other) = default;

    void GetCompressedComplex(size_t uncompressed_index, T &outX, T &outY) const {
        auto runOneIter = [&](auto &zx, auto &zy) {
            auto zx_old = zx;
            zx = zx * zx - zy * zy + results.m_OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2 } * zx_old * zy + results.m_OrbitYLow;
            HdrReduce(zy);
            };

        if (CachedIter1.UncompressedIter == uncompressed_index) {
            outX = CachedIter1.zx;
            outY = CachedIter1.zy;
            return;
        }

        if (CachedIter2.UncompressedIter == uncompressed_index) {
            outX = CachedIter2.zx;
            outY = CachedIter2.zy;
            return;
        }

        auto LinearScan = [&](CachedIter<T> &iter, CachedIter<T> &other) -> bool {
            if (iter.UncompressedIter + 1 == uncompressed_index) {
                bool condition =
                    (iter.CompressedIter + 1 < results.m_FullOrbit.GetSize() &&
                        (iter.UncompressedIter + 1 <
                            results.m_FullOrbit[iter.CompressedIter + 1].u.f.CompressionIndex)) ||
                    iter.CompressedIter + 1 == results.m_FullOrbit.GetSize();

                if (condition) {
                    other = iter;
                    runOneIter(iter.zx, iter.zy);
                    iter.UncompressedIter = uncompressed_index;
                } else {
                    other = iter;
                    iter.CompressedIter++;
                    iter.zx = results.m_FullOrbit[iter.CompressedIter].x;
                    iter.zy = results.m_FullOrbit[iter.CompressedIter].y;
                    iter.UncompressedIter = results.m_FullOrbit[iter.CompressedIter].u.f.CompressionIndex;
                }

                return true;
            }

            return false;
            };

        bool ret = LinearScan(CachedIter1, CachedIter2);
        if (ret) {
            outX = CachedIter1.zx;
            outY = CachedIter1.zy;
            return;
        }

        ret = LinearScan(CachedIter2, CachedIter1);
        if (ret) {
            outX = CachedIter2.zx;
            outY = CachedIter2.zy;
            return;
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

                if (results.m_FullOrbit[mid].u.f.CompressionIndex < uncompressed_index) {
                    low = mid + 1;
                } else if (results.m_FullOrbit[mid].u.f.CompressionIndex > uncompressed_index) {
                    high = mid - 1;
                } else {
                    return mid;
                }
            }

            return high;
            };


        auto BestCompressedIndexGuess = BinarySearch(uncompressed_index);

        T zx = results.m_FullOrbit[BestCompressedIndexGuess].x;
        T zy = results.m_FullOrbit[BestCompressedIndexGuess].y;

        for (size_t cur_uncompressed_index = results.m_FullOrbit[BestCompressedIndexGuess].u.f.CompressionIndex;
            cur_uncompressed_index < results.m_UncompressedItersInOrbit;
            cur_uncompressed_index++) {

            if (cur_uncompressed_index == uncompressed_index) {
                CachedIter2 = CachedIter1;
                CachedIter1 = CachedIter<T>(zx, zy, cur_uncompressed_index, BestCompressedIndexGuess);
                outX = zx;
                outY = zy;
                return;
            }

            runOneIter(zx, zy);
        }

        outX = {};
        outY = {};
        return;
    }

    template<class U>
    HDRFloatComplex<U> GetCompressedComplex(size_t uncompressed_index) const {
        T outX, outY;
        GetCompressedComplex(uncompressed_index, outX, outY);
        return { outX, outY };
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

    const PerturbationResults<IterType, T, PExtras> &results;
    mutable CachedIter<T> CachedIter1;
    mutable CachedIter<T> CachedIter2;
};

#ifndef __CUDACC__

template<typename IterType, class T, PerturbExtras PExtras>
class IntermediateRuntimeDecompressor : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    IntermediateRuntimeDecompressor(const PerturbationResults<IterType, T, PExtras> &results) :
        results(results),
        m_CachedIter{},
        cx{},
        cy{},
        Two{},
        Zero{},
        Temp{} {
        mpf_init2(cx, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set(cx, *results.GetHiX().backendRaw());

        mpf_init2(cy, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set(cy, *results.GetHiY().backendRaw());

        mpf_init2(Two, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set_d(Two, 2.0);

        mpf_init2(Zero, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set_d(Zero, 0.0);

        for (auto &temp : Temp) {
            mpf_init2(temp, AuthoritativeReuseExtraPrecisionInBits);
        }
    }

    IntermediateRuntimeDecompressor(const IntermediateRuntimeDecompressor &other) = default;

    ~IntermediateRuntimeDecompressor() {
        mpf_clear(cx);
        mpf_clear(cy);
        mpf_clear(Two);
        mpf_clear(Zero);

        for (auto &temp : Temp) {
            mpf_clear(temp);
        }
    }

    void GetReuseEntries(
        size_t uncompressed_index,
        const mpf_t *&outX,
        const mpf_t *&outY) const {

        auto runOneIter = [&](
            mpf_t zx,
            mpf_t zy) {

                // auto zx_old = zx;
                // zx = zx * zx - zy * zy + cx;
                // zy = Two * zx_old * zy + cy;

                mpf_set(Temp[0], zx);

                mpf_mul(Temp[1], zx, zx);
                mpf_mul(Temp[2], zy, zy);
                mpf_sub(Temp[3], Temp[1], Temp[2]);
                mpf_add(zx, Temp[3], cx);

                mpf_mul(Temp[4], Temp[0], zy);
                mpf_mul(Temp[4], Temp[4], Two);
                mpf_add(zy, Temp[4], cy);
            };

        if (m_CachedIter.UncompressedIter == uncompressed_index) {
            outX = &m_CachedIter.zx;
            outY = &m_CachedIter.zy;
            return;
        }

        if (uncompressed_index == 0) {
            mpf_set(m_CachedIter.zx, Zero);
            mpf_set(m_CachedIter.zy, Zero);
            m_CachedIter.UncompressedIter = 0;
            m_CachedIter.CompressedIter = 0;

            outX = &m_CachedIter.zx;
            outY = &m_CachedIter.zy;
            return;
        }

        auto LinearScan = [&](CachedIter &iter) -> void {
            assert(iter.UncompressedIter + 1 == uncompressed_index);
            assert(
                results.m_ReuseX.size() == results.m_ReuseY.size() &&
                results.m_ReuseX.size() == results.m_ReuseIndices.size());

            bool condition =
                (iter.CompressedIter + 1 < results.m_ReuseX.size() &&
                    (iter.UncompressedIter + 1 <
                        results.m_ReuseIndices[iter.CompressedIter + 1])) ||
                iter.CompressedIter + 1 == results.m_ReuseX.size();

            if (condition) {
                runOneIter(iter.zx, iter.zy);
                iter.UncompressedIter = uncompressed_index;
            } else {
                iter.CompressedIter++;
                mpf_set(iter.zx, *results.m_ReuseX[iter.CompressedIter].backendRaw());
                mpf_set(iter.zy, *results.m_ReuseY[iter.CompressedIter].backendRaw());
                iter.UncompressedIter = results.m_ReuseIndices[iter.CompressedIter];
            }
            };

        LinearScan(m_CachedIter);
        outX = &m_CachedIter.zx;
        outY = &m_CachedIter.zy;
    }

private:
    struct CachedIter {
        CachedIter() :
            zx{},
            zy{},
            UncompressedIter{ UINT64_MAX - 1 },
            CompressedIter{ UINT64_MAX - 1 } {

            mpf_init2(zx, AuthoritativeReuseExtraPrecisionInBits);
            mpf_init2(zy, AuthoritativeReuseExtraPrecisionInBits);
        }

        ~CachedIter() {
            mpf_clear(zx);
            mpf_clear(zy);
        }

        CachedIter &operator=(const CachedIter &other) = delete;
        CachedIter(const CachedIter &other) = delete;
        CachedIter(CachedIter &&other) noexcept = delete;
        CachedIter &operator=(CachedIter &&other) noexcept = delete;

        mpf_t zx;
        mpf_t zy;
        IterTypeFull UncompressedIter;
        IterTypeFull CompressedIter;
    };

    const PerturbationResults<IterType, T, PExtras> &results;
    mutable CachedIter m_CachedIter;

    mpf_t cx;
    mpf_t cy;
    mpf_t Two;
    mpf_t Zero;
    mutable mpf_t Temp[5];
};

template<typename IterType, class T, PerturbExtras PExtras>
class IntermediateMaxRuntimeDecompressor : public TemplateHelpers<IterType, T, PExtras> {
public:
    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    IntermediateMaxRuntimeDecompressor(const PerturbationResults<IterType, T, PExtras> &results) :
        results(results),
        m_CachedIter{},
        cx{},
        cy{},
        Two{},
        Zero{},
        Temp{} {
        mpf_init2(cx, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set(cx, *results.GetHiX().backendRaw());

        mpf_init2(cy, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set(cy, *results.GetHiY().backendRaw());

        mpf_init2(Two, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set_d(Two, 2.0);

        mpf_init2(Zero, AuthoritativeReuseExtraPrecisionInBits);
        mpf_set_d(Zero, 0.0);

        for (auto &temp : Temp) {
            mpf_init2(temp, AuthoritativeReuseExtraPrecisionInBits);
        }
    }

    IntermediateMaxRuntimeDecompressor(const IntermediateMaxRuntimeDecompressor &other) = default;

    ~IntermediateMaxRuntimeDecompressor() {
        mpf_clear(cx);
        mpf_clear(cy);
        mpf_clear(Two);
        mpf_clear(Zero);

        for (auto &temp : Temp) {
            mpf_clear(temp);
        }
    }

    void GetReuseEntries(
        size_t uncompressed_index,
        const mpf_t *&outX,
        const mpf_t *&outY) const {

        auto runOneIter = [&](
            mpf_t zx,
            mpf_t zy) {

                // auto zx_old = zx;
                // zx = zx * zx - zy * zy + cx;
                // zy = Two * zx_old * zy + cy;

                mpf_set(Temp[0], zx);

                mpf_mul(Temp[1], zx, zx);
                mpf_mul(Temp[2], zy, zy);
                mpf_sub(Temp[3], Temp[1], Temp[2]);
                mpf_add(zx, Temp[3], cx);

                mpf_mul(Temp[4], Temp[0], zy);
                mpf_mul(Temp[4], Temp[4], Two);
                mpf_add(zy, Temp[4], cy);
            };

        if (m_CachedIter.UncompressedIter == uncompressed_index) {
            outX = &m_CachedIter.zx;
            outY = &m_CachedIter.zy;
            return;
        }

        if (uncompressed_index == 0) {
            mpf_set(m_CachedIter.zx, Zero);
            mpf_set(m_CachedIter.zy, Zero);
            m_CachedIter.UncompressedIter = 0;
            m_CachedIter.CompressedIter = 0;

            outX = &m_CachedIter.zx;
            outY = &m_CachedIter.zy;
            return;
        }

        auto LinearScan = [&](CachedIter &iter) -> void {
            assert(iter.UncompressedIter + 1 == uncompressed_index);
            assert(
                results.m_ReuseX.size() == results.m_ReuseY.size() &&
                results.m_ReuseX.size() == results.m_ReuseIndices.size());

            bool condition =
                (iter.CompressedIter + 1 < results.m_ReuseX.size() &&
                    (iter.UncompressedIter + 1 <
                        results.m_ReuseIndices[iter.CompressedIter + 1])) ||
                iter.CompressedIter + 1 == results.m_ReuseX.size();

            if (condition) {
                runOneIter(iter.zx, iter.zy);
                iter.UncompressedIter = uncompressed_index;
            } else {
                iter.CompressedIter++;
                mpf_set(iter.zx, *results.m_ReuseX[iter.CompressedIter].backendRaw());
                mpf_set(iter.zy, *results.m_ReuseY[iter.CompressedIter].backendRaw());
                iter.UncompressedIter = results.m_ReuseIndices[iter.CompressedIter];
            }
            };

        LinearScan(m_CachedIter);
        outX = &m_CachedIter.zx;
        outY = &m_CachedIter.zy;
    }

private:
    struct CachedIter {
        CachedIter() :
            zx{},
            zy{},
            UncompressedIter{ UINT64_MAX - 1 },
            CompressedIter{ UINT64_MAX - 1 } {

            mpf_init2(zx, AuthoritativeReuseExtraPrecisionInBits);
            mpf_init2(zy, AuthoritativeReuseExtraPrecisionInBits);
        }

        ~CachedIter() {
            mpf_clear(zx);
            mpf_clear(zy);
        }

        CachedIter &operator=(const CachedIter &other) = delete;
        CachedIter(const CachedIter &other) = delete;
        CachedIter(CachedIter &&other) noexcept = delete;
        CachedIter &operator=(CachedIter &&other) noexcept = delete;

        mpf_t zx;
        mpf_t zy;
        IterTypeFull UncompressedIter;
        IterTypeFull CompressedIter;
    };

    const PerturbationResults<IterType, T, PExtras> &results;
    mutable CachedIter m_CachedIter;

    mpf_t cx;
    mpf_t cy;
    mpf_t Two;
    mpf_t Zero;
    mutable mpf_t Temp[5];
};

#endif