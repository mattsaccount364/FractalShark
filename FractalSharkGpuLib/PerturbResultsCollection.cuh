//////////////////////////////////////////////////////////////////////////////

PerturbResultsCollection::PerturbResultsCollection() {
    m_Results1.Init();
    m_Results2.Init();
}

PerturbResultsCollection::~PerturbResultsCollection() {
    DeleteAllInternal(m_Results1);
    DeleteAllInternal(m_Results2);
}

template<typename Type, PerturbExtras PExtras>
void PerturbResultsCollection::SetPtr32(
    size_t GenerationNumber,
    InternalResults &Results,
    GPUPerturbSingleResults<uint32_t, Type, PExtras> *ptr) {

    if (Results.m_GenerationNumber == GenerationNumber) {
        return;
    }

    Results.m_GenerationNumber = GenerationNumber;

    if constexpr (std::is_same<Type, float>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32FloatDisable;
            Results.m_Results32FloatDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32FloatEnable;
            Results.m_Results32FloatEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32FloatRC;
            Results.m_Results32FloatRC = ptr;
        }
    } else if constexpr (std::is_same<Type, double>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32DoubleDisable;
            Results.m_Results32DoubleDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32DoubleEnable;
            Results.m_Results32DoubleEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32DoubleRC;
            Results.m_Results32DoubleRC = ptr;
        }
    } else if constexpr (std::is_same<Type, CudaDblflt<dblflt>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32CudaDblfltDisable;
            Results.m_Results32CudaDblfltDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32CudaDblfltEnable;
            Results.m_Results32CudaDblfltEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32CudaDblfltRC;
            Results.m_Results32CudaDblfltRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32HdrFloatDisable;
            Results.m_Results32HdrFloatDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32HdrFloatEnable;
            Results.m_Results32HdrFloatEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32HdrFloatRC;
            Results.m_Results32HdrFloatRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32HdrDoubleDisable;
            Results.m_Results32HdrDoubleDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32HdrDoubleEnable;
            Results.m_Results32HdrDoubleEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32HdrDoubleRC;
            Results.m_Results32HdrDoubleRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results32HdrCudaMattDblfltDisable;
            Results.m_Results32HdrCudaMattDblfltDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results32HdrCudaMattDblfltEnable;
            Results.m_Results32HdrCudaMattDblfltEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results32HdrCudaMattDblfltRC;
            Results.m_Results32HdrCudaMattDblfltRC = ptr;
        }
    }
}

template<typename Type, PerturbExtras PExtras>
void PerturbResultsCollection::SetPtr64(
    size_t GenerationNumber,
    InternalResults &Results,
    GPUPerturbSingleResults<uint64_t, Type, PExtras> *ptr) {

    if (Results.m_GenerationNumber == GenerationNumber) {
        return;
    }

    Results.m_GenerationNumber = GenerationNumber;

    if constexpr (std::is_same<Type, float>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64FloatDisable;
            Results.m_Results64FloatDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64FloatEnable;
            Results.m_Results64FloatEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64FloatRC;
            Results.m_Results64FloatRC = ptr;
        }
    } else if constexpr (std::is_same<Type, double>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64DoubleDisable;
            Results.m_Results64DoubleDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64DoubleEnable;
            Results.m_Results64DoubleEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64DoubleRC;
            Results.m_Results64DoubleRC = ptr;
        }
    } else if constexpr (std::is_same<Type, CudaDblflt<dblflt>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64CudaDblfltDisable;
            Results.m_Results64CudaDblfltDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64CudaDblfltEnable;
            Results.m_Results64CudaDblfltEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64CudaDblfltRC;
            Results.m_Results64CudaDblfltRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64HdrFloatDisable;
            Results.m_Results64HdrFloatDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64HdrFloatEnable;
            Results.m_Results64HdrFloatEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64HdrFloatRC;
            Results.m_Results64HdrFloatRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64HdrDoubleDisable;
            Results.m_Results64HdrDoubleDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64HdrDoubleEnable;
            Results.m_Results64HdrDoubleEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64HdrDoubleRC;
            Results.m_Results64HdrDoubleRC = ptr;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            delete Results.m_Results64HdrCudaMattDblfltDisable;
            Results.m_Results64HdrCudaMattDblfltDisable = ptr;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            delete Results.m_Results64HdrCudaMattDblfltEnable;
            Results.m_Results64HdrCudaMattDblfltEnable = ptr;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            delete Results.m_Results64HdrCudaMattDblfltRC;
            Results.m_Results64HdrCudaMattDblfltRC = ptr;
        }
    }
}

template<typename IterType, typename Type, PerturbExtras PExtras>
void PerturbResultsCollection::SetPtr1(size_t GenerationNumber1, GPUPerturbSingleResults<IterType, Type, PExtras> *ptr) {
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        SetPtr32<Type, PExtras>(GenerationNumber1, m_Results1, ptr);
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        SetPtr64<Type, PExtras>(GenerationNumber1, m_Results1, ptr);
    }
}

template<typename IterType, typename Type, PerturbExtras PExtras>
void PerturbResultsCollection::SetPtr2(size_t GenerationNumber2, GPUPerturbSingleResults<IterType, Type, PExtras> *ptr) {
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        SetPtr32<Type, PExtras>(GenerationNumber2, m_Results2, ptr);
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        SetPtr64<Type, PExtras>(GenerationNumber2, m_Results2, ptr);
    }
}

template<typename Type, typename SubType>
void PerturbResultsCollection::SetLaReferenceInternal32(
    size_t LaGenerationNumber,
    InternalResults &Results,
    GPU_LAReference<uint32_t, Type, SubType> *LaReference) {

    if (LaGenerationNumber == Results.m_LaGenerationNumber) {
        return;
    }

    Results.m_LaGenerationNumber = LaGenerationNumber;

    if constexpr (std::is_same<Type, float>::value) {
        delete Results.m_LaReference32Float;
        Results.m_LaReference32Float = LaReference;
    } else if constexpr (std::is_same<Type, double>::value) {
        delete Results.m_LaReference32Double;
        Results.m_LaReference32Double = LaReference;
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        delete Results.m_LaReference32CudaDblflt;
        Results.m_LaReference32CudaDblflt = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        delete Results.m_LaReference32HdrFloat;
        Results.m_LaReference32HdrFloat = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        delete Results.m_LaReference32HdrDouble;
        Results.m_LaReference32HdrDouble = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        delete Results.m_LaReference32HdrCudaMattDblflt;
        Results.m_LaReference32HdrCudaMattDblflt = LaReference;
    }
}

template<typename Type, typename SubType>
void PerturbResultsCollection::SetLaReferenceInternal64(
    size_t LaGenerationNumber,
    InternalResults &Results,
    GPU_LAReference<uint64_t, Type, SubType> *LaReference) {

    if (LaGenerationNumber == Results.m_LaGenerationNumber) {
        return;
    }

    Results.m_LaGenerationNumber = LaGenerationNumber;

    if constexpr (std::is_same<Type, float>::value) {
        delete Results.m_LaReference64Float;
        Results.m_LaReference64Float = LaReference;
    } else if constexpr (std::is_same<Type, double>::value) {
        delete Results.m_LaReference64Double;
        Results.m_LaReference64Double = LaReference;
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        delete Results.m_LaReference64CudaDblflt;
        Results.m_LaReference64CudaDblflt = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        delete Results.m_LaReference64HdrFloat;
        Results.m_LaReference64HdrFloat = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        delete Results.m_LaReference64HdrDouble;
        Results.m_LaReference64HdrDouble = LaReference;
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        delete Results.m_LaReference64HdrCudaMattDblflt;
        Results.m_LaReference64HdrCudaMattDblflt = LaReference;
    }
}

template<typename IterType, typename Type, typename SubType>
void PerturbResultsCollection::SetLaReferenceInternal(
    size_t LaGenerationNumber,
    InternalResults &Results,
    GPU_LAReference<IterType, Type, SubType> *LaReference) {
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        SetLaReferenceInternal32<Type, SubType>(LaGenerationNumber, Results, LaReference);
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        SetLaReferenceInternal64<Type, SubType>(LaGenerationNumber, Results, LaReference);
    }
}

template<typename IterType, typename Type, typename SubType>
void PerturbResultsCollection::SetLaReference1(
    size_t LaGenerationNumber,
    GPU_LAReference<IterType, Type, SubType> *LaReference) {
    SetLaReferenceInternal(LaGenerationNumber, m_Results1, LaReference);
}

template<typename Type, PerturbExtras PExtras>
GPUPerturbSingleResults<uint32_t, Type, PExtras> *PerturbResultsCollection::GetPtrInternal32(InternalResults &Results) {
    if constexpr (std::is_same<Type, float>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32FloatDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32FloatEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32FloatRC;
        }
    } else if constexpr (std::is_same<Type, double>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32DoubleDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32DoubleEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32DoubleRC;
        }
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32CudaDblfltDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32CudaDblfltEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32CudaDblfltRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32HdrFloatDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32HdrFloatEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32HdrFloatRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32HdrDoubleDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32HdrDoubleEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32HdrDoubleRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results32HdrCudaMattDblfltDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results32HdrCudaMattDblfltEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results32HdrCudaMattDblfltRC;
        }
    }
}

template<typename Type, PerturbExtras PExtras>
GPUPerturbSingleResults<uint64_t, Type, PExtras> *PerturbResultsCollection::GetPtrInternal64(InternalResults &Results) {
    if constexpr (std::is_same<Type, float>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64FloatDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64FloatEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64FloatRC;
        }
    } else if constexpr (std::is_same<Type, double>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64DoubleDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64DoubleEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64DoubleRC;
        }
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64CudaDblfltDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64CudaDblfltEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64CudaDblfltRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64HdrFloatDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64HdrFloatEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64HdrFloatRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64HdrDoubleDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64HdrDoubleEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64HdrDoubleRC;
        }
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return Results.m_Results64HdrCudaMattDblfltDisable;
        } else if constexpr (PExtras == PerturbExtras::Bad) {
            return Results.m_Results64HdrCudaMattDblfltEnable;
        } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            return Results.m_Results64HdrCudaMattDblfltRC;
        }
    }
}

template<typename IterType, typename Type, PerturbExtras PExtras>
GPUPerturbSingleResults<IterType, Type, PExtras> *PerturbResultsCollection::GetPtrInternal(InternalResults &Results) {
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        return GetPtrInternal32<Type, PExtras>(Results);
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        return GetPtrInternal64<Type, PExtras>(Results);
    }
}

template<typename Type, typename SubType>
GPU_LAReference<uint32_t, Type, SubType> *PerturbResultsCollection::GetLaReferenceInternal32(
    InternalResults &Results) {
    if constexpr (std::is_same<Type, float>::value) {
        return Results.m_LaReference32Float;
    } else if constexpr (std::is_same<Type, double>::value) {
        return Results.m_LaReference32Double;
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        return Results.m_LaReference32CudaDblflt;
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        return Results.m_LaReference32HdrFloat;
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        return Results.m_LaReference32HdrDouble;
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        return Results.m_LaReference32HdrCudaMattDblflt;
    }
}

template<typename Type, typename SubType>
GPU_LAReference<uint64_t, Type, SubType> *PerturbResultsCollection::GetLaReferenceInternal64(
    InternalResults &Results) {
    if constexpr (std::is_same<Type, float>::value) {
        return Results.m_LaReference64Float;
    } else if constexpr (std::is_same<Type, double>::value) {
        return Results.m_LaReference64Double;
    } else if constexpr (std::is_same<Type, CudaDblflt<MattDblflt>>::value) {
        return Results.m_LaReference64CudaDblflt;
    } else if constexpr (std::is_same<Type, HDRFloat<float>>::value) {
        return Results.m_LaReference64HdrFloat;
    } else if constexpr (std::is_same<Type, HDRFloat<double>>::value) {
        return Results.m_LaReference64HdrDouble;
    } else if constexpr (std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        return Results.m_LaReference64HdrCudaMattDblflt;
    }
}

template<typename IterType, typename Type, typename SubType>
GPU_LAReference<IterType, Type, SubType> *PerturbResultsCollection::GetLaReferenceInternal(
    InternalResults &Results) {
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        return GetLaReferenceInternal32<Type, SubType>(Results);
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        return GetLaReferenceInternal64<Type, SubType>(Results);
    }
}

template<typename IterType, typename Type, PerturbExtras PExtras>
GPUPerturbSingleResults<IterType, Type, PExtras> *PerturbResultsCollection::GetPtr1() {
    return GetPtrInternal<IterType, Type, PExtras>(m_Results1);
}

template<typename IterType, typename Type, PerturbExtras PExtras>
GPUPerturbSingleResults<IterType, Type, PExtras> *PerturbResultsCollection::GetPtr2() {
    return GetPtrInternal<IterType, Type, PExtras>(m_Results2);
}

size_t PerturbResultsCollection::GetHostGenerationNumber1() const {
    return m_Results1.m_GenerationNumber;
}

size_t PerturbResultsCollection::GetHostGenerationNumber2() const {
    return m_Results2.m_GenerationNumber;
}

size_t PerturbResultsCollection::GetHostLaGenerationNumber1() const {
    return m_Results1.m_LaGenerationNumber;
}

size_t PerturbResultsCollection::GetHostLaGenerationNumber2() const {
    return m_Results2.m_LaGenerationNumber;
}

template<typename IterType, typename Type, typename SubType>
GPU_LAReference<IterType, Type, SubType> *PerturbResultsCollection::GetLaReference1() {
    return GetLaReferenceInternal<IterType, Type, SubType>(m_Results1);
}

template<typename IterType, typename Type, typename SubType>
GPU_LAReference<IterType, Type, SubType> *PerturbResultsCollection::GetLaReference2() {
    return GetLaReferenceInternal<IterType, Type, SubType>(m_Results2);
}

void PerturbResultsCollection::DeleteAllInternal(InternalResults &Results) {
    Results.m_GenerationNumber = 0;
    Results.m_LaGenerationNumber = 0;

    if (Results.m_Results32FloatDisable) {
        delete Results.m_Results32FloatDisable;
        Results.m_Results32FloatDisable = nullptr;
    }
    if (Results.m_Results32FloatEnable) {
        delete Results.m_Results32FloatEnable;
        Results.m_Results32FloatEnable = nullptr;
    }
    if (Results.m_Results32FloatRC) {
        delete Results.m_Results32FloatRC;
        Results.m_Results32FloatRC = nullptr;
    }
    if (Results.m_Results32DoubleDisable) {
        delete Results.m_Results32DoubleDisable;
        Results.m_Results32DoubleDisable = nullptr;
    }
    if (Results.m_Results32DoubleEnable) {
        delete Results.m_Results32DoubleEnable;
        Results.m_Results32DoubleEnable = nullptr;
    }
    if (Results.m_Results32DoubleRC) {
        delete Results.m_Results32DoubleRC;
        Results.m_Results32DoubleRC = nullptr;
    }
    if (Results.m_Results32CudaDblfltDisable) {
        delete Results.m_Results32CudaDblfltDisable;
        Results.m_Results32CudaDblfltDisable = nullptr;
    }
    if (Results.m_Results32CudaDblfltEnable) {
        delete Results.m_Results32CudaDblfltEnable;
        Results.m_Results32CudaDblfltEnable = nullptr;
    }
    if (Results.m_Results32CudaDblfltRC) {
        delete Results.m_Results32CudaDblfltRC;
        Results.m_Results32CudaDblfltRC = nullptr;
    }

    if (Results.m_Results32HdrFloatDisable) {
        delete Results.m_Results32HdrFloatDisable;
        Results.m_Results32HdrFloatDisable = nullptr;
    }
    if (Results.m_Results32HdrFloatEnable) {
        delete Results.m_Results32HdrFloatEnable;
        Results.m_Results32HdrFloatEnable = nullptr;
    }
    if (Results.m_Results32HdrFloatRC) {
        delete Results.m_Results32HdrFloatRC;
        Results.m_Results32HdrFloatRC = nullptr;
    }
    if (Results.m_Results32HdrDoubleDisable) {
        delete Results.m_Results32HdrDoubleDisable;
        Results.m_Results32HdrDoubleDisable = nullptr;
    }
    if (Results.m_Results32HdrDoubleEnable) {
        delete Results.m_Results32HdrDoubleEnable;
        Results.m_Results32HdrDoubleEnable = nullptr;
    }
    if (Results.m_Results32HdrDoubleRC) {
        delete Results.m_Results32HdrDoubleRC;
        Results.m_Results32HdrDoubleRC = nullptr;
    }
    if (Results.m_Results32HdrCudaMattDblfltDisable) {
        delete Results.m_Results32HdrCudaMattDblfltDisable;
        Results.m_Results32HdrCudaMattDblfltDisable = nullptr;
    }
    if (Results.m_Results32HdrCudaMattDblfltEnable) {
        delete Results.m_Results32HdrCudaMattDblfltEnable;
        Results.m_Results32HdrCudaMattDblfltEnable = nullptr;
    }
    if (Results.m_Results32HdrCudaMattDblfltRC) {
        delete Results.m_Results32HdrCudaMattDblfltRC;
        Results.m_Results32HdrCudaMattDblfltRC = nullptr;
    }

    /////////

    if (Results.m_Results64FloatDisable) {
        delete Results.m_Results64FloatDisable;
        Results.m_Results64FloatDisable = nullptr;
    }
    if (Results.m_Results64FloatEnable) {
        delete Results.m_Results64FloatEnable;
        Results.m_Results64FloatEnable = nullptr;
    }
    if (Results.m_Results64FloatRC) {
        delete Results.m_Results64FloatRC;
        Results.m_Results64FloatRC = nullptr;
    }
    if (Results.m_Results64DoubleDisable) {
        delete Results.m_Results64DoubleDisable;
        Results.m_Results64DoubleDisable = nullptr;
    }
    if (Results.m_Results64DoubleEnable) {
        delete Results.m_Results64DoubleEnable;
        Results.m_Results64DoubleEnable = nullptr;
    }
    if (Results.m_Results64DoubleRC) {
        delete Results.m_Results64DoubleRC;
        Results.m_Results64DoubleRC = nullptr;
    }
    if (Results.m_Results64CudaDblfltDisable) {
        delete Results.m_Results64CudaDblfltDisable;
        Results.m_Results64CudaDblfltDisable = nullptr;
    }
    if (Results.m_Results64CudaDblfltEnable) {
        delete Results.m_Results64CudaDblfltEnable;
        Results.m_Results64CudaDblfltEnable = nullptr;
    }
    if (Results.m_Results64CudaDblfltRC) {
        delete Results.m_Results64CudaDblfltRC;
        Results.m_Results64CudaDblfltRC = nullptr;
    }

    if (Results.m_Results64HdrFloatDisable) {
        delete Results.m_Results64HdrFloatDisable;
        Results.m_Results64HdrFloatDisable = nullptr;
    }
    if (Results.m_Results64HdrFloatEnable) {
        delete Results.m_Results64HdrFloatEnable;
        Results.m_Results64HdrFloatEnable = nullptr;
    }
    if (Results.m_Results64HdrFloatRC) {
        delete Results.m_Results64HdrFloatRC;
        Results.m_Results64HdrFloatRC = nullptr;
    }
    if (Results.m_Results64HdrDoubleDisable) {
        delete Results.m_Results64HdrDoubleDisable;
        Results.m_Results64HdrDoubleDisable = nullptr;
    }
    if (Results.m_Results64HdrDoubleEnable) {
        delete Results.m_Results64HdrDoubleEnable;
        Results.m_Results64HdrDoubleEnable = nullptr;
    }
    if (Results.m_Results64HdrDoubleRC) {
        delete Results.m_Results64HdrDoubleRC;
        Results.m_Results64HdrDoubleRC = nullptr;
    }
    if (Results.m_Results64HdrCudaMattDblfltDisable) {
        delete Results.m_Results64HdrCudaMattDblfltDisable;
        Results.m_Results64HdrCudaMattDblfltDisable = nullptr;
    }
    if (Results.m_Results64HdrCudaMattDblfltEnable) {
        delete Results.m_Results64HdrCudaMattDblfltEnable;
        Results.m_Results64HdrCudaMattDblfltEnable = nullptr;
    }
    if (Results.m_Results64HdrCudaMattDblfltRC) {
        delete Results.m_Results64HdrCudaMattDblfltRC;
        Results.m_Results64HdrCudaMattDblfltRC = nullptr;
    }

    ////////

    if (Results.m_LaReference32Float) {
        delete Results.m_LaReference32Float;
        Results.m_LaReference32Float = nullptr;
    }

    if (Results.m_LaReference32Double) {
        delete Results.m_LaReference32Double;
        Results.m_LaReference32Double = nullptr;
    }

    if (Results.m_LaReference32CudaDblflt) {
        delete Results.m_LaReference32CudaDblflt;
        Results.m_LaReference32CudaDblflt = nullptr;
    }

    if (Results.m_LaReference32HdrFloat) {
        delete Results.m_LaReference32HdrFloat;
        Results.m_LaReference32HdrFloat = nullptr;
    }

    if (Results.m_LaReference32HdrDouble) {
        delete Results.m_LaReference32HdrDouble;
        Results.m_LaReference32HdrDouble = nullptr;
    }

    if (Results.m_LaReference32HdrCudaMattDblflt) {
        delete Results.m_LaReference32HdrCudaMattDblflt;
        Results.m_LaReference32HdrCudaMattDblflt = nullptr;
    }

    ////////

    if (Results.m_LaReference64Float) {
        delete Results.m_LaReference64Float;
        Results.m_LaReference64Float = nullptr;
    }

    if (Results.m_LaReference64Double) {
        delete Results.m_LaReference64Double;
        Results.m_LaReference64Double = nullptr;
    }

    if (Results.m_LaReference64CudaDblflt) {
        delete Results.m_LaReference64CudaDblflt;
        Results.m_LaReference64CudaDblflt = nullptr;
    }

    if (Results.m_LaReference64HdrFloat) {
        delete Results.m_LaReference64HdrFloat;
        Results.m_LaReference64HdrFloat = nullptr;
    }

    if (Results.m_LaReference64HdrDouble) {
        delete Results.m_LaReference64HdrDouble;
        Results.m_LaReference64HdrDouble = nullptr;
    }

    if (Results.m_LaReference64HdrCudaMattDblflt) {
        delete Results.m_LaReference64HdrCudaMattDblflt;
        Results.m_LaReference64HdrCudaMattDblflt = nullptr;
    }
}

void PerturbResultsCollection::DeleteAll() {
    DeleteAllInternal(m_Results1);
    DeleteAllInternal(m_Results2);
}
