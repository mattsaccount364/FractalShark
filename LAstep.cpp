#include "stdafx.h"
#include "LAstep.h"
#include "LAInfoDeep.h"

LAstep::HDRFloatComplex LAstep::Evaluate(HDRFloatComplex DeltaSub0) {
    return LAjdeep->Evaluate(newDzDeep, DeltaSub0);
}

LAstep::HDRFloatComplex LAstep::EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc(z, dzdc);
}

LAstep::HDRFloatComplex LAstep::EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return LAjdeep->EvaluateDzdc2(z, dzdc2, dzdc);
}

LAstep::HDRFloatComplex LAstep::getZ(HDRFloatComplex DeltaSubN) {
    return Refp1Deep.plus(DeltaSubN);
}
