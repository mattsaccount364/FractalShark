#pragma once

#define InstantiateIsPerturbationResultUsefulHere(T, PExtras) \
template bool RefOrbitCalc::IsPerturbationResultUsefulHere<uint32_t, T, false, PExtras>(size_t i); \
template bool RefOrbitCalc::IsPerturbationResultUsefulHere<uint64_t, T, false, PExtras>(size_t i); \


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

InstantiateIsPerturbationResultUsefulHere(double, PerturbExtras::EnableCompression);
InstantiateIsPerturbationResultUsefulHere(float, PerturbExtras::EnableCompression);
InstantiateIsPerturbationResultUsefulHere(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<double>, PerturbExtras::EnableCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<float>, PerturbExtras::EnableCompression);
InstantiateIsPerturbationResultUsefulHere(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);



//////////////////////////////////////////////

template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Disable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::BenchmarkMode::Enable>();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//#define InstantiateGetAndCreateUsefulPerturbationResults(IterTypeT, T, SubTypeT, PExtras, RefOrbitExtras) \
//    template PerturbationResults<IterTypeT, T, PExtras>* \
//    RefOrbitCalc::GetAndCreateUsefulPerturbationResults< \
//        IterTypeT, \
//        T, \
//        SubTypeT, \
//        PExtras, \
//        RefOrbitExtras>();

#define InstantiateGetAndCreateUsefulPerturbationResults2(IterTypeT, T, SubType, PExtras, Ex, ConvertTType) \
    template PerturbationResults<IterTypeT, ConvertTType, PExtras>* \
    RefOrbitCalc::GetAndCreateUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex, ConvertTType>(float CompressionError);

#define InstantiateGetAndCreateUsefulPerturbationResults1(IterTypeT, T, SubType, PExtras, Ex) \
    template PerturbationResults<IterTypeT, T, PExtras>* \
    RefOrbitCalc::GetAndCreateUsefulPerturbationResults<IterTypeT, T, SubType, PExtras, Ex>(float CompressionError);


InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, double, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, float, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<double>, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<float>, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, float, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<float>, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2, CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2, HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, float, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2, CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint32_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2, HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint32_t, HDRFloat<float>, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, double, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, float, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<double>, double, PerturbExtras::Bad, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<float>, float, PerturbExtras::Bad, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, float, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<float>, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::None);

InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, float, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t, double, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2, CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2, HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<float>, float, PerturbExtras::Disable, RefOrbitCalc::Extras::IncludeLAv2);

InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, float, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t, double, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2, CudaDblflt<MattDblflt>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);
InstantiateGetAndCreateUsefulPerturbationResults2(uint64_t, HDRFloat<double>, double, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2, HDRFloat<CudaDblflt<MattDblflt>>);
InstantiateGetAndCreateUsefulPerturbationResults1(uint64_t, HDRFloat<float>, float, PerturbExtras::EnableCompression, RefOrbitCalc::Extras::IncludeLAv2);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
template PerturbationResults<uint32_t, float, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    double,
    PerturbExtras::Bad,
    float,
    PerturbExtras::Bad>(PerturbationResults<uint32_t, double, PerturbExtras::Bad>&);
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    PerturbExtras::Bad,
    HDRFloat<float>,
    PerturbExtras::Bad>(PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad>&);
template PerturbationResults<uint32_t, float, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    HDRFloat<float>,
    PerturbExtras::Bad,
    float,
    PerturbExtras::Bad>(PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad>&);

template PerturbationResults<uint32_t, float, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    double,
    PerturbExtras::Disable,
    float,
    PerturbExtras::Disable>(PerturbationResults<uint32_t, double, PerturbExtras::Disable>&);
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    PerturbExtras::Disable,
    HDRFloat<float>,
    PerturbExtras::Disable>(PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable>&);
template PerturbationResults<uint32_t, float, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint32_t,
    HDRFloat<float>,
    PerturbExtras::Disable,
    float,
    PerturbExtras::Disable>(PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>&);
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
template PerturbationResults<uint64_t, float, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    double,
    PerturbExtras::Bad,
    float,
    PerturbExtras::Bad>(PerturbationResults<uint64_t, double, PerturbExtras::Bad>&);
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    PerturbExtras::Bad,
    HDRFloat<float>,
    PerturbExtras::Bad>(PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad>&);
template PerturbationResults<uint64_t, float, PerturbExtras::Bad>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    HDRFloat<float>,
    PerturbExtras::Bad,
    float,
    PerturbExtras::Bad>(PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad>&);

template PerturbationResults<uint64_t, float, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    double,
    PerturbExtras::Disable,
    float,
    PerturbExtras::Disable>(PerturbationResults<uint64_t, double, PerturbExtras::Disable>&);
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    PerturbExtras::Disable,
    HDRFloat<float>,
    PerturbExtras::Disable>(PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable>&);
template PerturbationResults<uint64_t, float, PerturbExtras::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    uint64_t,
    HDRFloat<float>,
    PerturbExtras::Disable,
    float,
    PerturbExtras::Disable>(PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>&);
///////////////////////////////////////////////////////////////////

