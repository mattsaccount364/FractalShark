#ifndef __GDD_GQF_INLINE_CU__
#define __GDD_GQF_INLINE_CU__

namespace GQF {

	// TODO mrenz busted.  2^11 + 1???????
//#define _QF_SPLITTER            (8193.0f)                         // = 2^13 + 1
#define _QF_SPLITTER            (4097.0f)                         // = 2^12 + 1
//#define _QF_SPLITTER            (2049.0f)                         // = 2^11 + 1
//#define _QF_SPLITTER            (1025.0f)                         // = 2^10 + 1

#define _QF_SPLIT_THRESH        (2.076918743413931051412198531688e+34f)         // = 2^114


//#define _QD_SPLITTER            (134217729.0)                   // = 2^27 + 1
//#define _QD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996


/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|
	__host__ __device__
		float quick_two_sum(float a, float b, float& err)
	{

		//if (b == 0.0f) {
		//	err = 0.0f;
		//	return (a + b);
		//}

		float s = a + b;
		err = b - (s - a);

		return s;
	}

	__host__ __device__
		float two_sum(float a, float b, float& err)
	{

		//if ((a == 0.0f) || (b == 0.0f)) {
		//	err = 0.0f;
		//	return (a + b);
		//}

		float s = a + b;
		float bb = s - a;
		err = (a - (s - bb)) + (b - bb);

		return s;
	}


	//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
	__host__ __device__
		float quick_two_diff(float a, float b, float& err)
	{
		//if (a == b) {
		//	err = 0.0f;
		//	return 0.0f;
		//}

		float s;

		/*
		if(fabs((a-b)/a) < GPU_D_EPS) {
					s = 0.0f;
					err = 0.0f;
					return s;
			}
		*/

		s = a - b;
		err = (a - s) - b;
		return s;
	}

	//computes fl( a - b ) and err( a - b )
	__host__ __device__
		float two_diff(float a, float b, float& err)
	{
		//if (a == b) {
		//	err = 0.0f;
		//	return 0.0f;
		//}

		float s = a - b;

		/*
		if(fabs((a-b)/a) < GPU_D_EPS) {
			s = 0.0f;
			err = 0.0f;
			return s;
		}
		*/

		float bb = s - a;
		err = (a - (s - bb)) - (b + bb);
		return s;
	}


	//// Computes high word and lo word of a 
	//__host__ __device__
	//	void split(double a, double& hi, double& lo)
	//{
	//	double temp;
	//	if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH)
	//	{
	//		a *= 3.7252902984619140625e-09;  // 2^-28
	//		temp = _QD_SPLITTER * a;
	//		hi = temp - (temp - a);
	//		lo = a - hi;
	//		hi *= 268435456.0;          // 2^28
	//		lo *= 268435456.0;          // 2^28
	//	}
	//	else {
	//		temp = _QD_SPLITTER * a;
	//		hi = temp - (temp - a);
	//		lo = a - hi;
	//	}
	//}

	// Computes high word and lo word of a 
	__host__ __device__
		void split(float a, float& hi, float& lo)
	{
		//// TODO WRONG WRONG WRONG
		//float temp;
		//if (a > _QF_SPLIT_THRESH || a < -_QF_SPLIT_THRESH)
		//{
		//	//a *= 6.103515625e-5f; // 2^-14
		//	a *= 1.220703125e-4f; // 2^-13
		//	//a *= 2.44140625e-4f; // 2^-12

		//	temp = _QF_SPLITTER * a;
		//	hi = temp - (temp - a);
		//	lo = a - hi;
		//	hi *= 8192.0f; // 2^13
		//	lo *= 8192.0f;
		//} else 	{
		//	temp = _QF_SPLITTER * a;
		//	hi = temp - (temp - a);
		//	lo = a - hi;
		//}

		const float split = 4097.0f;//(1<<12)+1;
		float t = a*split;
		hi = t-(t-a);
		lo = a-hi;
	}

	/* Computes fl(a*b) and err(a*b). */
	__device__
		float two_prod(float a, float b, float& err)
	{

		float a_hi, a_lo, b_hi, b_lo;
		float p = a * b;
		split(a, a_hi, a_lo);
		split(b, b_hi, b_lo);

		//err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 
		err = (a_hi * b_hi) - p + (a_hi * b_lo) + (a_lo * b_hi) + (a_lo * b_lo);

		return p;
	}

	/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
	__host__ __device__
		float two_sqr(float a, float& err)
	{
		float hi, lo;
		float q = a * a;
		split(a, hi, lo);
		err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
		return q;
	}

	/* Computes the nearest integer to d. */
	__host__ __device__
		float nint(float d)
	{
		if (d == floor(d))
			return d;
		return floor(d + 0.5f);
	}

}

#endif /* __GDD_GQF_INLINE_CU__ */
