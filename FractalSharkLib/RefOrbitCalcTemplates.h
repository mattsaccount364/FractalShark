#pragma once

template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, float, float, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, double, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Disable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, float, float, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, double, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint32_t, HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Enable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, float, float, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, double, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Disable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, float, float, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, double, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<uint64_t, HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Enable>();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template PerturbationResults<uint32_t, double, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    double,
    double,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, float, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    float,
    float,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();

template PerturbationResults<uint32_t, double, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, float, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    float,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();

///////////

template PerturbationResults<uint32_t, double, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint32_t, float, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    float,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2,
    CudaDblflt<MattDblflt>>();
template PerturbationResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2,
    HDRFloat<CudaDblflt<MattDblflt>>>();
template PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint32_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template PerturbationResults<uint64_t, double, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    double,
    double,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, float, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    float,
    float,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Bad,
    RefOrbitCalc::Extras::None>();

template PerturbationResults<uint64_t, double, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, float, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    float,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::None>();

///////////

template PerturbationResults<uint64_t, double, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint64_t, float, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    float,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    double,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2,
    CudaDblflt<MattDblflt>>();
template PerturbationResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<double>,
    double,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2,
    HDRFloat<CudaDblflt<MattDblflt>>>();
template PerturbationResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    uint64_t,
    HDRFloat<float>,
    float,
    PerturbExtras::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
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

