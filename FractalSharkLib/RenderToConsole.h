// RenderToConsole — ASCII/ANSI terminal rendering of fractal iteration data.
//
// Reads the iteration grid from a Fractal that has already been computed
// (CalcFractal(true) must have been called), block-averages down to console
// dimensions, and writes the result to an ostream.

#pragma once

#include <cstddef>
#include <iosfwd>

class Fractal;

struct ConsoleRenderOptions {
    size_t ConsoleWidth = 80;
    size_t ConsoleHeight = 24;
    bool Color = false; // true → ANSI 256-color escape codes
};

// Renders the current iteration grid as ASCII art.  The Fractal must have
// been rendered via CalcFractal(true) so that m_CurIters is populated.
void RenderToConsole(const Fractal &fractal, const ConsoleRenderOptions &opts, std::ostream &out);
