#ifndef HEAT_LUT_HPP
#define HEAT_LUT_HPP

#include <cstdint>

struct rgba8_t {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
	std::uint8_t a;
};

rgba8_t heat_lut(float x);


#endif //HEAT_LUT_HPP