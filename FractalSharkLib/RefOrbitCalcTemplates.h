#pragma once

#define InstantiateIsPerturbationResultUsefulHere(T, PExtras)                                           \
    template bool RefOrbitCalc::IsPerturbationResultUsefulHere<uint32_t, T, false, PExtras>(size_t i)   \
        const;                                                                                          \
    template bool RefOrbitCalc::IsPerturbationResultUsefulHere<uint64_t, T, false, PExtras>(size_t i)   \
        const;

InstantiateIsPerturbationResultUsefulHere(double, PerturbExtras::Disable);
InstantiateIsPerturbationResultUsefulHere(float, PerturbExtras::Disable);
InstantiateIsPerturbationResultUsefulHere(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<double>, PerturbExtras::Disable);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<float>, PerturbExtras::Disable);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);

InstantiateIsPerturbationResultUsefulHere(double, PerturbExtras::Bad);
InstantiateIsPerturbationResultUsefulHere(float, PerturbExtras::Bad);
InstantiateIsPerturbationResultUsefulHere(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<double>, PerturbExtras::Bad);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<float>, PerturbExtras::Bad);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);

InstantiateIsPerturbationResultUsefulHere(double, PerturbExtras::SimpleCompression);
InstantiateIsPerturbationResultUsefulHere(float, PerturbExtras::SimpleCompression);
InstantiateIsPerturbationResultUsefulHere(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<double>, PerturbExtras::SimpleCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<float>, PerturbExtras::SimpleCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<CudaDblflt<MattDblflt>>,
                                          PerturbExtras::SimpleCompression);

//////////////////////////////////////////////

#define InstantiateAddPerturbationReferencePoint(IterTypeT, T, SubTypeT, PExtras, RefOrbitExtras)       \
    template void                                                                                       \
    RefOrbitCalc::AddPerturbationReferencePoint<IterTypeT, T, SubTypeT, PExtras, RefOrbitExtras>(const PointZoomBBConverter &);

InstantiateAddPerturbationReferencePoint(
    uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);

InstantiateAddPerturbationReferencePoint(
    uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);

InstantiateAddPerturbationReferencePoint(
    uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable);

InstantiateAddPerturbationReferencePoint(
    uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);
InstantiateAddPerturbationReferencePoint(
    uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define InstantiateGetAndCreateUsefulPerturbationResults(IterTypeT, T, SubTypeT, PExtras, RefOrbitExtras) \
//    template PerturbationResults<IterTypeT, T, PExtras>* \
//    RefOrbitCalc::GetAndCreateUsefulPerturbationResults< \
//        IterTypeT, \
//        T, \
//        SubTypeT, \
//        PExtras, \
//        RefOrbitExtras>();

#define InstantiateGetAndCreateUsefulPerturbationResults2(                                              \
    IterTypeT, T, SubType, PExtras, Ex, ConvertTType)                                                   \
    template PerturbationResults<IterTypeT, ConvertTType, PExtras> *RefOrbitCalc::                      \
        GetAndCreateUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex, ConvertTType>(        \
            const PointZoomBBConverter &);                                                              \
    template const PerturbationResults<IterTypeT, ConvertTType, PExtras> *                              \
    RefOrbitCalc::GetUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex, ConvertTType>()      \
        const;

#define InstantiateGetAndCreateUsefulPerturbationResults1(IterTypeT, T, SubType, PExtras, Ex)           \
    template PerturbationResults<IterTypeT, T, PExtras> *                                               \
    RefOrbitCalc::GetAndCreateUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex>(            \
        const PointZoomBBConverter &);                                                                  \
    template const PerturbationResults<IterTypeT, T, PExtras> *                                         \
    RefOrbitCalc::GetUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex>() const;

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, double, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, float, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<double>, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<float>, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::None,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::None,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, double, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::None,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, float, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<double>, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::None,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, double, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint32_t, float, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t,
                                                  HDRFloat<float>,
                                                  float,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, double, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, float, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<double>, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<float>, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::None,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::None,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, double, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::None,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, float, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<double>, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::None,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<float>, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::Disable,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, double, double, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  double,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(
    uint64_t, float, float, PerturbExtras::SimpleCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t,
                                                  HDRFloat<double>,
                                                  double,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2,
                                                  HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t,
                                                  HDRFloat<float>,
                                                  float,
                                                  PerturbExtras::SimpleCompression,
                                                  RefOrbitCalc::Extras::IncludeLAv2);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
template PerturbationResults<uint32_t, float, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t, double, PerturbExtras::Bad, float, PerturbExtras::Bad>(
        PerturbationResults<uint32_t, double, PerturbExtras::Bad> &);
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t,
                                  HDRFloat<double>,
                                  PerturbExtras::Bad,
                                  HDRFloat<float>,
                                  PerturbExtras::Bad>(
        PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad> &);
template PerturbationResults<uint32_t, float, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t,
                                  HDRFloat<float>,
                                  PerturbExtras::Bad,
                                  float,
                                  PerturbExtras::Bad>(
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad> &);

template PerturbationResults<uint32_t, float, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t,
                                  double,
                                  PerturbExtras::Disable,
                                  float,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint32_t, double, PerturbExtras::Disable> &);
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t,
                                  HDRFloat<double>,
                                  PerturbExtras::Disable,
                                  HDRFloat<float>,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable> &);
template PerturbationResults<uint32_t, float, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint32_t,
                                  HDRFloat<float>,
                                  PerturbExtras::Disable,
                                  float,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> &);
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
template PerturbationResults<uint64_t, float, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t, double, PerturbExtras::Bad, float, PerturbExtras::Bad>(
        PerturbationResults<uint64_t, double, PerturbExtras::Bad> &);
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t,
                                  HDRFloat<double>,
                                  PerturbExtras::Bad,
                                  HDRFloat<float>,
                                  PerturbExtras::Bad>(
        PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad> &);
template PerturbationResults<uint64_t, float, PerturbExtras::Bad> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t,
                                  HDRFloat<float>,
                                  PerturbExtras::Bad,
                                  float,
                                  PerturbExtras::Bad>(
        PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad> &);

template PerturbationResults<uint64_t, float, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t,
                                  double,
                                  PerturbExtras::Disable,
                                  float,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint64_t, double, PerturbExtras::Disable> &);
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t,
                                  HDRFloat<double>,
                                  PerturbExtras::Disable,
                                  HDRFloat<float>,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable> &);
template PerturbationResults<uint64_t, float, PerturbExtras::Disable> *RefOrbitCalc::
    CopyUsefulPerturbationResults<uint64_t,
                                  HDRFloat<float>,
                                  PerturbExtras::Disable,
                                  float,
                                  PerturbExtras::Disable>(
        PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable> &);
///////////////////////////////////////////////////////////////////
