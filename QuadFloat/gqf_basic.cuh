#ifndef __GQF_BASIC_CU__
#define __GQF_BASIC_CU__


#include "../QuadFloat/common.cuh"

namespace GQF {

	/** normalization functions */
	__host__ __device__
		void quick_renorm(float& c0, float& c1,
			float& c2, float& c3, float& c4)
	{
		float t0, t1, t2, t3;
		float s;
		s = quick_two_sum(c3, c4, t3);
		s = quick_two_sum(c2, s, t2);
		s = quick_two_sum(c1, s, t1);
		c0 = quick_two_sum(c0, s, t0);

		s = quick_two_sum(t2, t3, t2);
		s = quick_two_sum(t1, s, t1);
		c1 = quick_two_sum(t0, s, t0);

		s = quick_two_sum(t1, t2, t1);
		c2 = quick_two_sum(t0, s, t0);

		c3 = t0 + t1;
	}

	__host__ __device__
		void renorm(float& c0, float& c1,
			float& c2, float& c3)
	{
		float s0, s1, s2 = 0.0f, s3 = 0.0f;

		//if (isinf(c0)) return;

		s0 = quick_two_sum(c2, c3, c3);
		s0 = quick_two_sum(c1, s0, c2);
		c0 = quick_two_sum(c0, s0, c1);

		s0 = c0;
		s1 = c1;
		//if (s1 != 0.0f) {
			s1 = quick_two_sum(s1, c2, s2);
			//if (s2 != 0.0f)
				s2 = quick_two_sum(s2, c3, s3);
		//	else
		//		s1 = quick_two_sum(s1, c3, s2);
		//}
		//else {
		//	s0 = quick_two_sum(s0, c2, s1);
		//	if (s1 != 0.0f)
		//		s1 = quick_two_sum(s1, c3, s2);
		//	else
		//		s0 = quick_two_sum(s0, c3, s1);
		//}

		c0 = s0;
		c1 = s1;
		c2 = s2;
		c3 = s3;
	}

	__host__ __device__
		void renorm(float& c0, float& c1,
			float& c2, float& c3, float& c4)
	{
		float s0, s1, s2 = 0.0f, s3 = 0.0f;

		//if (isinf(c0)) return;

		s0 = quick_two_sum(c3, c4, c4);
		s0 = quick_two_sum(c2, s0, c3);
		s0 = quick_two_sum(c1, s0, c2);
		c0 = quick_two_sum(c0, s0, c1);

		s0 = c0;
		s1 = c1;

		s0 = quick_two_sum(c0, c1, s1);
		//if (s1 != 0.0f)
		//{
			s1 = quick_two_sum(s1, c2, s2);
			//if (s2 != 0.0f) {
				s2 = quick_two_sum(s2, c3, s3);
				//if (s3 != 0.0f)
					s3 += c4;
		//		else
		//			s2 += c4;
		//	}
		//	else {
		//		s1 = quick_two_sum(s1, c3, s2);
		//		if (s2 != 0.0f)
		//			s2 = quick_two_sum(s2, c4, s3);
		//		else
		//			s1 = quick_two_sum(s1, c4, s2);
		//	}
		//}
		//else {
		//	s0 = quick_two_sum(s0, c2, s1);
		//	if (s1 != 0.0f) {
		//		s1 = quick_two_sum(s1, c3, s2);
		//		if (s2 != 0.0f)
		//			s2 = quick_two_sum(s2, c4, s3);
		//		else
		//			s1 = quick_two_sum(s1, c4, s2);
		//	}
		//	else {
		//		s0 = quick_two_sum(s0, c3, s1);
		//		if (s1 != 0.0f)
		//			s1 = quick_two_sum(s1, c4, s2);
		//		else
		//			s0 = quick_two_sum(s0, c4, s1);
		//	}
		//}

		c0 = s0;
		c1 = s1;
		c2 = s2;
		c3 = s3;
	}

	__host__ __device__
		void renorm(gqf_real& x) {
		renorm(x.x, x.y, x.z, x.w);
	}

	__host__ __device__
		void renorm(gqf_real& x, float& e) {
		renorm(x.x, x.y, x.z, x.w, e);
	}

	/** additions */
	__host__ __device__
		void three_sum(float& a, float& b, float& c)
	{
		float t1, t2, t3;
		t1 = two_sum(a, b, t2);
		a = two_sum(c, t1, t3);
		b = two_sum(t2, t3, c);
	}

	__host__ __device__
		void three_sum2(float& a, float& b, float& c) {
		float t1, t2, t3;
		t1 = two_sum(a, b, t2);
		a = two_sum(c, t1, t3);
		b = (t2 + t3);
	}

	///qd = qd + float
	__host__ __device__
		gqf_real operator+(const gqf_real& a, float b) {
		float c0, c1, c2, c3;
		float e;

		c0 = two_sum(a.x, b, e);
		c1 = two_sum(a.y, e, e);
		c2 = two_sum(a.z, e, e);
		c3 = two_sum(a.w, e, e);

		renorm(c0, c1, c2, c3, e);

		return make_qf(c0, c1, c2, c3);
	}

	///qd = float + qd
	__host__ __device__
		gqf_real operator+(float a, const gqf_real& b)
	{
		return (b + a);
	}

	///qd = qd + qd
	__host__ __device__
		gqf_real sloppy_add(const gqf_real& a, const gqf_real& b)
	{
		float s0, s1, s2, s3;
		float t0, t1, t2, t3;

		float v0, v1, v2, v3;
		float u0, u1, u2, u3;
		float w0, w1, w2, w3;

		s0 = a.x + b.x;
		s1 = a.y + b.y;
		s2 = a.z + b.z;
		s3 = a.w + b.w;

		v0 = s0 - a.x;
		v1 = s1 - a.y;
		v2 = s2 - a.z;
		v3 = s3 - a.w;

		u0 = s0 - v0;
		u1 = s1 - v1;
		u2 = s2 - v2;
		u3 = s3 - v3;

		w0 = a.x - u0;
		w1 = a.y - u1;
		w2 = a.z - u2;
		w3 = a.w - u3;

		u0 = b.x - v0;
		u1 = b.y - v1;
		u2 = b.z - v2;
		u3 = b.w - v3;

		t0 = w0 + u0;
		t1 = w1 + u1;
		t2 = w2 + u2;
		t3 = w3 + u3;

		s1 = two_sum(s1, t0, t0);
		three_sum(s2, t0, t1);
		three_sum2(s3, t0, t2);
		t0 = t0 + t1 + t3;

		renorm(s0, s1, s2, s3, t0);

		return make_qf(s0, s1, s2, s3);
	}

	__host__ __device__
		gqf_real operator+(const gqf_real& a, const gqf_real& b)
	{
		return sloppy_add(a, b);
	}


	/** subtractions */
	__host__ __device__
		gqf_real negative(const gqf_real& a)
	{
		return make_qf(-a.x, -a.y, -a.z, -a.w);
	}

	__host__ __device__
		gqf_real operator-(const gqf_real& a, float b)
	{
		return (a + (-b));
	}

	__host__ __device__
		gqf_real operator-(float a, const gqf_real& b)
	{
		return (a + negative(b));
	}

	__host__ __device__
		gqf_real operator-(const gqf_real& a, const gqf_real& b)
	{
		return (a + negative(b));
	}

	/** multiplications */
	__host__ __device__
		gqf_real mul_pwr2(const gqf_real& a, float b) {
		return make_qf(a.x * b, a.y * b, a.z * b, a.w * b);
	}


	//quad_double * float
	__device__
		gqf_real operator*(const gqf_real& a, float b)
	{
		float p0, p1, p2, p3;
		float q0, q1, q2;
		float s0, s1, s2, s3, s4;

		p0 = two_prod(a.x, b, q0);
		p1 = two_prod(a.y, b, q1);
		p2 = two_prod(a.z, b, q2);
		p3 = a.w * b;

		s0 = p0;

		s1 = two_sum(q0, p1, s2);

		three_sum(s2, q1, p2);

		three_sum2(q1, q2, p3);
		s3 = q1;

		s4 = q2 + p2;

		renorm(s0, s1, s2, s3, s4);
		return make_qf(s0, s1, s2, s3);
	}
	//quad_double = float*quad_double
	__device__
		gqf_real operator*(float a, const gqf_real& b)
	{
		return b * a;
	}

	__device__
		gqf_real sloppy_mul(const gqf_real& a, const gqf_real& b)
	{
		float p0, p1, p2, p3, p4, p5;
		float q0, q1, q2, q3, q4, q5;
		float t0, t1;
		float s0, s1, s2;

		p0 = two_prod(a.x, b.x, q0);

		p1 = two_prod(a.x, b.y, q1);
		p2 = two_prod(a.y, b.x, q2);

		p3 = two_prod(a.x, b.z, q3);
		p4 = two_prod(a.y, b.y, q4);
		p5 = two_prod(a.z, b.x, q5);


		/* Start Accumulation */
		three_sum(p1, p2, q0);

		//return make_qf(p1, p2, q0, 0.0f);

			/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
		three_sum(p2, q1, q2);
		three_sum(p3, p4, p5);
		/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
		s0 = two_sum(p2, p3, t0);
		s1 = two_sum(q1, p4, t1);
		s2 = q2 + p5;
		s1 = two_sum(s1, t0, t0);
		s2 += (t0 + t1);

		//return make_qf(s0, s1, t0, t1);

			/* O(eps^3) order terms */
			//!!!s1 = s1 + (a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x + q0 + q3 + q4 + q5);

		s1 = s1 + (__fmul_rn(a.x, b.w) + __fmul_rn(a.y, b.z) +
			__fmul_rn(a.z, b.y) + __fmul_rn(a.w, b.x) + q0 + q3 + q4 + q5);
		renorm(p0, p1, s0, s1, s2);

		return make_qf(p0, p1, s0, s1);

	}

	__device__
		gqf_real operator*(const gqf_real& a, const gqf_real& b) {
		return sloppy_mul(a, b);
	}

	__device__
		gqf_real sqr(const gqf_real& a)
	{
		float p0, p1, p2, p3, p4, p5;
		float q0, q1, q2, q3;
		float s0, s1;
		float t0, t1;

		p0 = two_sqr(a.x, q0);
		p1 = two_prod(2.0f * a.x, a.y, q1);
		p2 = two_prod(2.0f * a.x, a.z, q2);
		p3 = two_sqr(a.y, q3);

		p1 = two_sum(q0, p1, q0);

		q0 = two_sum(q0, q1, q1);
		p2 = two_sum(p2, p3, p3);

		s0 = two_sum(q0, p2, t0);
		s1 = two_sum(q1, p3, t1);

		s1 = two_sum(s1, t0, t0);
		t0 += t1;

		s1 = quick_two_sum(s1, t0, t0);
		p2 = quick_two_sum(s0, s1, t1);
		p3 = quick_two_sum(t1, t0, q0);

		p4 = 2.0f * a.x * a.w;
		p5 = 2.0f * a.y * a.z;

		p4 = two_sum(p4, p5, p5);
		q2 = two_sum(q2, q3, q3);

		t0 = two_sum(p4, q2, t1);
		t1 = t1 + p5 + q3;

		p3 = two_sum(p3, t0, p4);
		p4 = p4 + q0 + t1;

		renorm(p0, p1, p2, p3, p4);
		return make_qf(p0, p1, p2, p3);
	}

	/** divisions */
	__device__
		gqf_real sloppy_div(const gqf_real& a, const gqf_real& b)
	{
		float q0, q1, q2, q3;

		gqf_real r;

		q0 = a.x / b.x;
		r = a - (b * q0);

		q1 = r.x / b.x;
		r = r - (b * q1);

		q2 = r.x / b.x;
		r = r - (b * q2);

		q3 = r.x / b.x;

		renorm(q0, q1, q2, q3);

		return make_qf(q0, q1, q2, q3);
	}

	__device__
		gqf_real operator/(const gqf_real& a, const gqf_real& b)
	{
		return sloppy_div(a, b);
	}

	/* float / quad-float */
	__device__
		gqf_real operator/(float a, const gqf_real& b)
	{
		return make_qf(a) / b;
	}

	/* quad-float / float */
	__device__
		gqf_real operator/(const gqf_real& a, float b)
	{
		return a / make_qf(b);
	}

	/********** Miscellaneous **********/
	__host__ __device__
		gqf_real abs(const gqf_real& a)
	{
		return (a.x < 0.0f) ? (negative(a)) : (a);
	}

	/********************** Simple Conversion ********************/
	__host__ __device__
		float to_double(const gqf_real& a)
	{
		return a.x;
	}

	//__host__ __device__
	//	gqf_real ldexp(const gqf_real& a, int n)
	//{
	//	return make_qf(ldexp(a.x, n), ldexp(a.y, n),
	//		ldexp(a.z, n), ldexp(a.w, n));
	//}

	__device__
		gqf_real inv(const gqf_real& qd)
	{
		return 1.0f / qd;
	}


	/********** Greater-Than Comparison ***********/

	__host__ __device__
		bool operator>=(const gqf_real& a, const gqf_real& b)
	{
		return (a.x > b.x ||
			(a.x == b.x && (a.y > b.y ||
				(a.y == b.y && (a.z > b.z ||
					(a.z == b.z && a.w >= b.w))))));
	}

	/********** Greater-Than-Or-Equal-To Comparison **********/
	/*
	__device__
	bool operator>=(const gqf_real &a, float b) {
	  return (a.x > b || (a.x == b && a.y >= 0.0f));
	}

	__device__
	bool operator>=(float a, const gqf_real &b) {
	  return (b <= a);
	}

	__device__
	bool operator>=(const gqf_real &a, const gqf_real &b) {
	  return (a.x > b.x ||
			  (a.x == b.x && (a.y > b.y ||
								(a.y == b.y && (a.z > b.z ||
												  (a.z == b.z && a.w >= b.w))))));
	}
	*/

	/********** Less-Than Comparison ***********/
	__host__ __device__
		bool operator<(const gqf_real& a, float b) {
		return (a.x < b || (a.x == b && a.y < 0.0f));
	}

	__host__ __device__
		bool operator<(const gqf_real& a, const gqf_real& b) {
		return (a.x < b.x ||
			(a.x == b.x && (a.y < b.y ||
				(a.y == b.y && (a.z < b.z ||
					(a.z == b.z && a.w < b.w))))));
	}

	__host__ __device__
		bool operator<=(const gqf_real& a, const gqf_real& b) {
		return (a.x < b.x ||
			(a.x == b.x && (a.y < b.y ||
				(a.y == b.y && (a.z < b.z ||
					(a.z == b.z && a.w <= b.w))))));
	}

	__host__ __device__
		bool operator==(const gqf_real& a, const gqf_real& b) {
		return (a.x == b.x && a.y == b.y &&
			a.z == b.z && a.w == b.w);
	}



	/********** Less-Than-Or-Equal-To Comparison **********/
	__device__
		bool operator<=(const gqf_real& a, float b) {
		return (a.x < b || (a.x == b && a.y <= 0.0f));
	}

	/*
	__device__
	bool operator<=(float a, const gqf_real &b) {
		return (b >= a);
	}
	*/

	/*
	__device__
	bool operator<=(const gqf_real &a, const gqf_real &b) {
	  return (a.x < b.x ||
			  (a.x == b.x && (a.y < b.y ||
								(a.y == b.y && (a.z < b.z ||
												  (a.z == b.z && a.w <= b.w))))));
	}
	*/

	/********** Greater-Than-Or-Equal-To Comparison **********/
	__device__
		bool operator>=(const gqf_real& a, float b) {
		return (a.x > b || (a.x == b && a.y >= 0.0f));
	}

	__device__
		bool operator<=(float a, const gqf_real& b) {
		return (b >= a);
	}


	__device__
		bool operator>=(float a, const gqf_real& b) {
		return (b <= a);
	}


	/*
	__device__
	bool operator>=(const gqf_real &a, const gqf_real &b) {
	  return (a.x > b.x ||
			  (a.x == b.x && (a.y > b.y ||
								(a.y == b.y && (a.z > b.z ||
												  (a.z == b.z && a.w >= b.w))))));
	}

	*/

	/********** Greater-Than Comparison ***********/
	__host__ __device__
		bool operator>(const gqf_real& a, float b) {
		return (a.x > b || (a.x == b && a.y > 0.0f));
	}

	__host__ __device__
		bool operator<(float a, const gqf_real& b) {
		return (b > a);
	}

	__host__ __device__
		bool operator>(float a, const gqf_real& b) {
		return (b < a);
	}

	__host__ __device__
		bool operator>(const gqf_real& a, const gqf_real& b) {
		return (a.x > b.x ||
			(a.x == b.x && (a.y > b.y ||
				(a.y == b.y && (a.z > b.z ||
					(a.z == b.z && a.w > b.w))))));
	}


	__host__ __device__
		bool is_zero(const gqf_real& x)
	{
		return (x.x == 0.0f);
	}

	__host__ __device__
		bool is_one(const gqf_real& x)
	{
		return (x.x == 1.0f && x.y == 0.0f && x.z == 0.0f && x.w == 0.0f);
	}

	__host__ __device__
		bool is_positive(const gqf_real& x)
	{
		return (x.x > 0.0f);
	}

	__host__ __device__
		bool is_negative(const gqf_real& x)
	{
		return (x.x < 0.0f);
	}

	__device__
		gqf_real nint(const gqf_real& a) {
		float x0, x1, x2, x3;

		x0 = nint(a.x);
		x1 = x2 = x3 = 0.0f;

		if (x0 == a.x) {
			/* First float is already an integer. */
			x1 = nint(a.y);

			if (x1 == a.y) {
				/* Second float is already an integer. */
				x2 = nint(a.z);

				if (x2 == a.z) {
					/* Third float is already an integer. */
					x3 = nint(a.w);
				}
				else {
					if (fabs(x2 - a.z) == 0.5 && a.w < 0.0f) {
						x2 -= 1.0f;
					}
				}

			}
			else {
				if (fabs(x1 - a.y) == 0.5 && a.z < 0.0f) {
					x1 -= 1.0f;
				}
			}

		}
		else {
			/* First float is not an integer. */
			if (fabs(x0 - a.x) == 0.5 && a.y < 0.0f) {
				x0 -= 1.0f;
			}
		}

		renorm(x0, x1, x2, x3);
		return make_qf(x0, x1, x2, x3);
	}

	__device__
		gqf_real fabs(const gqf_real& a) {
		return abs(a);
	}

}

#endif


