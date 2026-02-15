#pragma once

#include "Fractal.h"

class AutoZoomer {
public:
    explicit AutoZoomer(Fractal &fractal);

    template <Fractal::AutoZoomHeuristic h>
    void Run();

private:
    Fractal &m_Fractal;
};
