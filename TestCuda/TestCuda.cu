#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#define MAX_LEN 2048  // Adjusted MAX_LEN to accommodate larger numbers

#define BLOCK_SIZE 256

// Struct to hold both integer and fractional parts of the high-precision number
struct HighPrecisionNumber {
    enum class CopyType {
        HostToHost,
        HostToDevice,
        DeviceToHost
    };

    enum class ConstructionType {
        Host,
        Device
    };

    HighPrecisionNumber();
    HighPrecisionNumber(ConstructionType type, uint32_t numDigits);
    HighPrecisionNumber(uint32_t numDigits);
    ~HighPrecisionNumber();
    HighPrecisionNumber(const HighPrecisionNumber &) = delete;
    HighPrecisionNumber(CopyType copyType, const HighPrecisionNumber&);
    HighPrecisionNumber& operator=(const HighPrecisionNumber&) = delete;

    std::string high_precision_to_string() const;

    uint32_t *Digits;
    uint32_t Size;        // Number of digits
    int32_t Exponent;    // Exponent in base 2^32
    bool IsNegative; // Sign of the number
    ConstructionType IsCudaMemory; // Flag to indicate if memory is allocated on the GPU
};

HighPrecisionNumber::HighPrecisionNumber()
    : Digits{},
    Size{},
    Exponent{},
    IsNegative{},
    IsCudaMemory{ConstructionType::Host} {
}

HighPrecisionNumber::HighPrecisionNumber(ConstructionType type, uint32_t)
    : Digits{},
    Size{},
    Exponent{},
    IsNegative{},
    IsCudaMemory{ConstructionType::Host} {
    if (type == ConstructionType::Host) {
        Digits = new uint32_t[Size];
        std::fill(Digits, Digits + Size, 0);
    } else {
        cudaMalloc(&Digits, Size * sizeof(uint32_t));
        cudaMemset(Digits, 0, Size * sizeof(uint32_t));
    }
}

HighPrecisionNumber::HighPrecisionNumber(uint32_t numDigits)
    : Digits{ new uint32_t[numDigits] },
    Size{ numDigits },
    Exponent{},
    IsNegative{},
    IsCudaMemory{ConstructionType::Host} {

    std::fill(Digits, Digits + numDigits, 0);
}

HighPrecisionNumber::HighPrecisionNumber(CopyType copyType, const HighPrecisionNumber &other)
    : Digits{},
    Size{ other.Size },
    Exponent{ other.Exponent },
    IsNegative{ other.IsNegative },
    IsCudaMemory{ ConstructionType::Host } {

    if (copyType == CopyType::HostToHost) {
        Digits = new uint32_t[Size];
        std::copy(other.Digits, other.Digits + Size, Digits);

        IsCudaMemory = ConstructionType::Host;
    } else if (copyType == CopyType::HostToDevice) {
        cudaMalloc(&Digits, Size * sizeof(uint32_t));
        cudaMemcpy(Digits, other.Digits, Size * sizeof(uint32_t), cudaMemcpyHostToDevice);

        IsCudaMemory = ConstructionType::Device;
    } else if (copyType == CopyType::DeviceToHost) {
        Digits = new uint32_t[Size];
        cudaMemcpy(Digits, other.Digits, Size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        IsCudaMemory = ConstructionType::Host;
    }
}

HighPrecisionNumber::~HighPrecisionNumber() {
    if (IsCudaMemory == ConstructionType::Device) {
        cudaFree(Digits);
    } else if (IsCudaMemory == ConstructionType::Host) {
        delete[] Digits;
    }
}

// Function to convert mpf_t to string
std::string mpf_tostring(mpf_t mpf_val) {
    char *str = NULL;
    gmp_asprintf(&str, "%.Ff", mpf_val);
    std::string result(str);
    free(str);
    return result;
}

// Function to convert mpf_t to HighPrecisionNumber
void mpf_to_high_precision(const mpf_t mpf_val, HighPrecisionNumber& number, int prec_bits) {
    // Clear existing digits if any
    if (number.Digits != nullptr) {
        delete[] number.Digits;
        number.Digits = nullptr;
    }

    // Get the absolute value of mpf_val
    mpf_t abs_val;
    mpf_init(abs_val);
    mpf_abs(abs_val, mpf_val);

    std::cout << "abs_val: " << mpf_tostring(abs_val) << std::endl;

    // Determine the sign
    number.IsNegative = (mpf_sgn(mpf_val) < 0);

    // Scale the number to convert fractional part to integer
    mpf_t scaled_val;
    mpf_init(scaled_val);
    mpf_mul_2exp(scaled_val, abs_val, prec_bits); // scaled_val = abs_val * 2^prec_bits

    std::cout << "scaled_val: " << mpf_tostring(scaled_val) << std::endl;

    // Convert scaled_val to an integer
    mpz_t int_val;
    mpz_init(int_val);
    mpz_set_f(int_val, scaled_val);

    // Check if int_val is zero
    if (mpz_cmp_ui(int_val, 0) == 0) {
        // The number is zero
        number.Size = 1;
        number.Digits = new uint32_t[1];
        number.Digits[0] = 0;
    } else {
        // Export mpz_t integer to uint32_t digits in base 2^32
        size_t count = 0;
        uint32_t* data = (uint32_t*)mpz_export(NULL, &count, -1, sizeof(uint32_t), 0, 0, int_val);

        // Allocate digits array
        number.Size = static_cast<int>(count);
        number.Digits = new uint32_t[number.Size];

        // Copy data into digits array
        for (size_t i = 0; i < count; ++i) {
            number.Digits[i] = data[i];
        }

        // Free the exported data
        free(data);
    }

    // Set the Exponent
    number.Exponent = -prec_bits;

    // Clean up
    mpz_clear(int_val);
    mpf_clear(scaled_val);
    mpf_clear(abs_val);
}

// CUDA kernel for high-precision subtraction with borrow handling
__global__ void subtract_high_precision_kernel(uint32_t *A, uint32_t *B, uint32_t *C, int len) {
    // Each thread subtracts one element of B from A
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ uint32_t s_borrow[BLOCK_SIZE]; // Shared memory for borrow

    uint64_t diff = 0;  // Change to uint64_t to hold values larger than 32 bits
    uint32_t borrow = 0;

    if (idx < len) {
        uint32_t a = A[idx];
        uint32_t b = B[idx];

        if (a >= b) {
            diff = (uint64_t)a - (uint64_t)b;
            borrow = 0;
        } else {
            diff = (uint64_t)a - (uint64_t)b + (1ULL << 32);  // Use 64-bit shift
            borrow = 1;
        }
    }

    // Store borrow into shared memory
    int thread_in_block = threadIdx.x;
    s_borrow[thread_in_block] = borrow;
    __syncthreads();

    // Perform borrow propagation using parallel prefix sum
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        uint32_t temp = 0;
        if (thread_in_block >= offset)
            temp = s_borrow[thread_in_block - offset];
        __syncthreads();
        if (thread_in_block >= offset)
            s_borrow[thread_in_block] += temp;
        __syncthreads();
    }

    // Adjust the difference with the propagated borrow
    if (idx < len) {
        if (s_borrow[thread_in_block] > 0) {
            if (diff >= s_borrow[thread_in_block]) {
                diff -= s_borrow[thread_in_block];
            } else {
                diff = diff - s_borrow[thread_in_block] + (1ULL << 32);
            }
        }
        C[idx] = (uint32_t)(diff & 0xFFFFFFFFULL);  // Store lower 32 bits

        // Handle borrow out of the block
        if (thread_in_block == blockDim.x - 1) {
            if (blockIdx.x < gridDim.x - 1) {
                atomicAdd(&C[(blockIdx.x + 1) * blockDim.x], -s_borrow[thread_in_block]);
            } else if (idx + 1 < len) {
                atomicAdd(&C[idx + 1], -s_borrow[thread_in_block]);
            }
        }
    }
}

// Should assume input arrays are of the same length
// TODO
// handle Exponent
__global__ void add_high_precision_kernel(
    const uint32_t *__restrict__ A,
    const uint32_t *__restrict__ B,
    uint32_t *C,
    int len,
    int32_t exponentA,
    int32_t exponentB,
    int32_t *resultExponent) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine the signs of A and B
    bool is_neg_a = (A[len - 1] & 0x80000000) != 0; // Check the most significant bit
    bool is_neg_b = (B[len - 1] & 0x80000000) != 0; // Check the most significant bit

    // Align exponents
    int exponentDiff = exponentA - exponentB;
    const uint32_t *largerExpDigits = A;
    const uint32_t *smallerExpDigits = B;
    int largerExp = exponentA;

    if (exponentDiff < 0) {
        // B has a larger exponent
        exponentDiff = -exponentDiff;
        largerExpDigits = B;
        smallerExpDigits = A;
        largerExp = exponentB;
    }

    // Initialize shared memory to hold the shifted value
    __shared__ uint32_t s_shifted[MAX_LEN];

    // Shift the digits of the smaller exponent to align with the larger exponent
    if (idx < len) {
        if (idx < exponentDiff) {
            s_shifted[idx] = 0; // Shift in zeros
        } else {
            s_shifted[idx] = smallerExpDigits[idx - exponentDiff];
        }
    }

    __syncthreads();

    uint32_t carry_in = 0;

    if (is_neg_a == is_neg_b) {
        // Both numbers have the same sign, perform addition
        if (idx < len) {
            uint32_t a_val = largerExpDigits[idx] & 0x7FFFFFFF; // Mask out the sign bit
            uint32_t b_val = s_shifted[idx] & 0x7FFFFFFF; // Mask out the sign bit
            uint64_t sum = (uint64_t)a_val + b_val + carry_in;

            // Store the result
            C[idx] = (uint32_t)(sum & 0xFFFFFFFF);

            // Calculate carry-out
            uint32_t carry_out = (uint32_t)(sum >> 32);

            // Propagate carry to the next position
            if (idx + 1 < len) {
                atomicAdd(&C[idx + 1], carry_out);
            }
        }
        __syncthreads();

        // Set the sign of the result
        if (idx == len - 1) {
            if (is_neg_a) {
                C[idx] |= 0x80000000; // Set the sign bit
            } else {
                C[idx] &= 0x7FFFFFFF; // Ensure the sign bit is not set
            }
        }
    } else {
        // Perform subtraction if signs are different
        // Determine which number is larger in magnitude
        bool a_is_larger = true;
        for (int i = len - 1; i >= 0; --i) {
            uint32_t a_val = largerExpDigits[i] & 0x7FFFFFFF; // Mask out the sign bit
            uint32_t b_val = s_shifted[i] & 0x7FFFFFFF; // Mask out the sign bit
            if (a_val < b_val) {
                a_is_larger = false;
                break;
            } else if (a_val > b_val) {
                break;
            }
        }

        // Perform A - B or B - A depending on which is larger
        uint32_t borrow_in = 0;
        if (idx < len) {
            uint32_t a_val = (a_is_larger ? largerExpDigits[idx] : s_shifted[idx]) & 0x7FFFFFFF; // Mask out the sign bit
            uint32_t b_val = (a_is_larger ? s_shifted[idx] : largerExpDigits[idx]) & 0x7FFFFFFF; // Mask out the sign bit

            uint64_t diff;
            if (a_val >= b_val + borrow_in) {
                diff = (uint64_t)a_val - b_val - borrow_in;
                borrow_in = 0;
            } else {
                diff = (uint64_t)a_val + (1ULL << 32) - b_val - borrow_in;
                borrow_in = 1;
            }

            // Store the result
            C[idx] = (uint32_t)(diff & 0xFFFFFFFF);

            // Propagate borrow to the next position
            if (idx + 1 < len) {
                atomicAdd(&C[idx + 1], -borrow_in);
            }
        }
        __syncthreads();

        // Set the sign of the result
        if (idx == len - 1) {
            if (a_is_larger) {
                C[idx] |= is_neg_a ? 0x80000000 : 0; // Set the sign bit based on the original A's sign
            } else {
                C[idx] |= is_neg_b ? 0x80000000 : 0; // Set the sign bit based on the original B's sign
            }
        }
    }

    // Store the exponent of the result
    if (idx == 0) {
        *resultExponent = largerExp; // Set the result's exponent to the larger exponent
    }
}

// CUDA kernel for high-precision multiplication
__global__ void multiply_high_precision_kernel(uint32_t *A, int lenA, uint32_t *B, int lenB, uint32_t *C) {
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    int idxB = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxA < lenA && idxB < lenB) {
        uint64_t product = (uint64_t)A[idxA] * (uint64_t)B[idxB];
        int idxC = idxA + idxB;

        // Atomically add lower and upper parts
        uint32_t lower = (uint32_t)(product & 0xFFFFFFFF);
        uint32_t upper = (uint32_t)(product >> 32);

        // Use atomic operations to avoid race conditions
        atomicAdd(&C[idxC], lower);

        // Handle carry
        uint32_t carry = upper;
        uint32_t temp = atomicAdd(&C[idxC], lower);
        if (temp + lower < temp) {
            carry++;
        }

        // Propagate carry
        int carry_idx = idxC + 1;
        while (carry > 0 && carry_idx < lenA + lenB) {
            uint32_t old = atomicAdd(&C[carry_idx], carry);
            if (old + carry < old) {
                carry = 1;
            } else {
                carry = 0;
            }
            carry_idx++;
        }
    }
}



void computeNextXY_gpu(
    HighPrecisionNumber &x,
    HighPrecisionNumber &y,
    HighPrecisionNumber &a,
    HighPrecisionNumber &b,
    int num_iter) {

    // Length of x, y, a, b are all the same.
    
    int blockSize = BLOCK_SIZE; // Number of threads per block
    auto hpLen = a.Size;
    auto gridSize = (hpLen + blockSize - 1) / blockSize;

    dim3 block(blockSize);

    // For multiplication, use 2D grids
    dim3 blockMul(16, 16); // Adjust block Size as needed
    dim3 gridMul((x.Size + blockMul.x - 1) / blockMul.x, (x.Size + blockMul.y - 1) / blockMul.y);

    // Allocate device memory for temporary variables
    uint32_t *d_x_squared;  int32_t *d_x_squared_exponent;
    uint32_t *d_y_squared;  int32_t *d_y_squared_exponent;
    uint32_t *d_xy;         int32_t *d_xy_exponent;
    uint32_t *d_two_xy;     int32_t *d_two_xy_exponent;
    uint32_t *d_temp_x;     int32_t *d_temp_x_exponent;
    uint32_t *d_temp_y;     int32_t *d_temp_y_exponent;

    cudaMalloc(&d_x_squared, hpLen * sizeof(uint32_t)); cudaMalloc(&d_x_squared_exponent, sizeof(int32_t));
    cudaMalloc(&d_y_squared, hpLen * sizeof(uint32_t)); cudaMalloc(&d_y_squared_exponent, sizeof(int32_t));
    cudaMalloc(&d_xy,        hpLen * sizeof(uint32_t)); cudaMalloc(&d_xy_exponent, sizeof(int32_t));
    cudaMalloc(&d_two_xy,    hpLen * sizeof(uint32_t)); cudaMalloc(&d_two_xy_exponent, sizeof(int32_t));
    cudaMalloc(&d_temp_x,    hpLen * sizeof(uint32_t)); cudaMalloc(&d_temp_x_exponent, sizeof(int32_t));
    cudaMalloc(&d_temp_y,    hpLen * sizeof(uint32_t)); cudaMalloc(&d_temp_y_exponent, sizeof(int32_t));

    // Initialize result arrays to zero
    cudaMemset(d_x_squared, 0, hpLen * sizeof(uint32_t)); cudaMemset(d_x_squared_exponent, 0, sizeof(int32_t));
    cudaMemset(d_y_squared, 0, hpLen * sizeof(uint32_t)); cudaMemset(d_y_squared_exponent, 0, sizeof(int32_t));
    cudaMemset(d_xy,        0, hpLen * sizeof(uint32_t)); cudaMemset(d_xy_exponent, 0, sizeof(int32_t));
    cudaMemset(d_two_xy,    0, hpLen * sizeof(uint32_t)); cudaMemset(d_two_xy_exponent, 0, sizeof(int32_t));
    cudaMemset(d_temp_x,    0, hpLen * sizeof(uint32_t)); cudaMemset(d_temp_x_exponent, 0, sizeof(int32_t));
    cudaMemset(d_temp_y,    0, hpLen * sizeof(uint32_t)); cudaMemset(d_temp_y_exponent, 0, sizeof(int32_t));

    for (int iter = 0; iter < num_iter; ++iter) {
        //// x_squared = x * x
        //multiply_high_precision_kernel << <gridMul, blockMul >> > (x.Digits, x.Size, x.Digits, x.Size, d_x_squared);
        //cudaDeviceSynchronize();

        //// y_squared = y * y
        //multiply_high_precision_kernel << <gridMul, blockMul >> > (y.Digits, y.Size, y.Digits, y.Size, d_y_squared);
        //cudaDeviceSynchronize();

        //// xy = x * y
        //multiply_high_precision_kernel << <gridMul, blockMul >> > (x.Digits, x.Size, y.Digits, y.Size, d_xy);
        //cudaDeviceSynchronize();

        //// two_xy = xy + xy
        //add_high_precision_kernel << <gridSize, block >> > (d_xy, d_xy, d_two_xy, size_xy);
        //cudaDeviceSynchronize();

        //// temp_x = x_squared - y_squared
        //subtract_high_precision_kernel << <gridSize, block >> > (d_x_squared, d_y_squared, d_temp_x, max(size_x_squared, size_y_squared));
        //cudaDeviceSynchronize();

        // temp_x = temp_x + a
        add_high_precision_kernel << <gridSize, block >> > (
            d_temp_x,
            a.Digits,
            d_temp_x,
            hpLen,
            x.Exponent,
            a.Exponent,
            d_temp_x_exponent);
        cudaDeviceSynchronize();

        // temp_y = two_xy + b
        add_high_precision_kernel << <gridSize, block >> > (
            d_two_xy,
            b.Digits,
            d_temp_y,
            hpLen,
            y.Exponent,
            b.Exponent,
            d_temp_y_exponent);
        cudaDeviceSynchronize();

        // Update x and y
        cudaMemcpy(x.Digits, d_temp_x, hpLen * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        x.Exponent = *d_temp_x_exponent;

        cudaMemcpy(y.Digits, d_two_xy, hpLen * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        y.Exponent = *d_two_xy_exponent;
    }

    // Free device memory
    cudaFree(d_x_squared); cudaFree(d_x_squared_exponent);
    cudaFree(d_y_squared); cudaFree(d_y_squared_exponent);
    cudaFree(d_xy);        cudaFree(d_xy_exponent);
    cudaFree(d_two_xy);    cudaFree(d_two_xy_exponent);
    cudaFree(d_temp_x);    cudaFree(d_temp_x_exponent);
    cudaFree(d_temp_y);    cudaFree(d_temp_y_exponent);
}

std::string HighPrecisionNumber::high_precision_to_string() const {
    // Initialize an mpz_t integer to hold the significand
    mpz_t mpz_value;
    mpz_init(mpz_value);

    // Import the digits array into the mpz_t integer
    // Note: mpz_import expects the least significant word first
    mpz_import(mpz_value, Size, -1, sizeof(uint32_t), 0, 0, Digits);

    // Adjust for sign
    if (IsNegative) {
        mpz_neg(mpz_value, mpz_value);
    }

    // Calculate the total exponent in bits
    // Since the exponent is in base 2^32, total_bits = exponent * 32
    int64_t total_bits = (int64_t)(Exponent); // Exponent is already in bits

    // Initialize an mpf_t floating-point number with sufficient precision
    mpf_t mpf_value;
    mp_bitcnt_t prec = mpz_sizeinbase(mpz_value, 2) + (total_bits > 0 ? total_bits : -total_bits) + 64;
    mpf_init2(mpf_value, prec);

    // Set mpf_value to mpz_value
    mpf_set_z(mpf_value, mpz_value);

    // Adjust the exponent by multiplying/dividing by 2^total_bits
    if (total_bits > 0) {
        mpf_mul_2exp(mpf_value, mpf_value, (mp_bitcnt_t)total_bits);
    } else if (total_bits < 0) {
        mpf_div_2exp(mpf_value, mpf_value, (mp_bitcnt_t)(-total_bits));
    }
    // If total_bits == 0, no adjustment is needed

    // Convert mpf_value to a decimal string
    // We use base 10 and let GMP decide the number of digits needed
    mp_exp_t decimal_exp;
    char *str = mpf_get_str(NULL, &decimal_exp, 10, 0, mpf_value);

    // Convert the mantissa string to a C++ string for easier manipulation
    std::string mantissa_str(str);
    free(str); // Free the string allocated by mpf_get_str

    // Determine where to place the decimal point
    size_t mantissa_len = mantissa_str.length();
    std::string result;

    if (IsNegative) {
        result = "-";
    }

    if (decimal_exp > 0) {
        if ((mp_exp_t)mantissa_len <= decimal_exp) {
            // The number is an integer; append zeros if necessary
            result += mantissa_str;
            result.append(decimal_exp - mantissa_len, '0');
        } else {
            // Insert decimal point within the mantissa string
            result += mantissa_str.substr(0, decimal_exp);
            result += '.';
            result += mantissa_str.substr(decimal_exp);
        }
    } else {
        // The number is less than 1; add leading zeros after "0."
        result += "0.";
        result.append(-decimal_exp, '0');
        result += mantissa_str;
    }

    // Clear GMP variables
    mpz_clear(mpz_value);
    mpf_clear(mpf_value);

    return result;
}

// Function to convert HighPrecisionNumber to mpf_t
void high_precision_to_mpf(const HighPrecisionNumber &hpNum, mpf_t &mpf_val) {
    // Initialize an mpz_t integer to hold the significand
    mpz_t mpz_value;
    mpz_init(mpz_value);

    // Import the digits array into the mpz_t integer
    // Note: mpz_import expects the least significant word first
    mpz_import(mpz_value, hpNum.Size, -1, sizeof(uint32_t), 0, 0, hpNum.Digits);

    // Adjust for sign
    if (hpNum.IsNegative) {
        mpz_neg(mpz_value, mpz_value);
    }

    // Set mpf_val to mpz_value
    mpf_set_z(mpf_val, mpz_value);

    // Adjust the exponent
    // Since the exponent is in base 2^32, adjust by multiplying/dividing by 2^(exponent * 32)
    int64_t total_bits = (int64_t)(hpNum.Exponent); // Exponent is already in bits
    if (total_bits > 0) {
        mpf_mul_2exp(mpf_val, mpf_val, (mp_bitcnt_t)total_bits);
    } else if (total_bits < 0) {
        mpf_div_2exp(mpf_val, mpf_val, (mp_bitcnt_t)(-total_bits));
    }
    // If exponent is zero, no adjustment needed

    // Clean up
    mpz_clear(mpz_value);
}

// Function to perform the calculation on the host using MPIR
void computeNextXY_host(mpf_t x, mpf_t y, mpf_t a, mpf_t b, int num_iter) {
    mpf_t x_squared, y_squared, two_xy, temp_x, temp_y;
    mpf_init(x_squared);
    mpf_init(y_squared);
    mpf_init(two_xy);
    mpf_init(temp_x);
    mpf_init(temp_y);

    for (int iter = 0; iter < num_iter; ++iter) {
        mpf_mul(x_squared, x, x); // x^2
        mpf_mul(y_squared, y, y); // y^2
        mpf_mul(temp_y, x, y);    // xy
        mpf_mul_ui(two_xy, temp_y, 2); // 2xy

        mpf_sub(temp_x, x_squared, y_squared); // x^2 - y^2
        mpf_add(temp_x, temp_x, a);            // x^2 - y^2 + a
        mpf_add(temp_y, two_xy, b);            // 2xy + b

        mpf_set(x, temp_x);
        mpf_set(y, temp_y);
    }

    mpf_clear(x_squared);
    mpf_clear(y_squared);
    mpf_clear(two_xy);
    mpf_clear(temp_x);
    mpf_clear(temp_y);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <x> <y> <a> <b>" << std::endl;
        return 1;
    }

    mpf_set_default_prec(1024);  // Set precision for MPIR floating point

    // Initialize MPIR variables
    mpf_t mpf_x, mpf_y, mpf_a, mpf_b;
    mpf_init(mpf_x);
    mpf_init(mpf_y);
    mpf_init(mpf_a);
    mpf_init(mpf_b);

    // Set values from command-line arguments
    mpf_set_str(mpf_x, argv[1], 10);
    mpf_set_str(mpf_y, argv[2], 10);
    mpf_set_str(mpf_a, argv[3], 10);
    mpf_set_str(mpf_b, argv[4], 10);

    // Make a copy of the input values for host computation
    mpf_t mpf_x_copy, mpf_y_copy;
    mpf_init(mpf_x_copy);
    mpf_init(mpf_y_copy);
    mpf_set(mpf_x_copy, mpf_x);
    mpf_set(mpf_y_copy, mpf_y);

    // Print the original input values
    std::cout << "Original input values:" << std::endl;
    std::cout << "X: " << mpf_tostring(mpf_x) << std::endl;
    std::cout << "Y: " << mpf_tostring(mpf_y) << std::endl;
    std::cout << "A: " << mpf_tostring(mpf_a) << std::endl;
    std::cout << "B: " << mpf_tostring(mpf_b) << std::endl;

    // Convert the input values to HighPrecisionNumber representations
    {
        HighPrecisionNumber x_num{};
        HighPrecisionNumber y_num{};
        HighPrecisionNumber a_num{};
        HighPrecisionNumber b_num{};

        mpf_to_high_precision(mpf_x, x_num, 256);
        mpf_to_high_precision(mpf_y, y_num, 256);
        mpf_to_high_precision(mpf_a, a_num, 256);
        mpf_to_high_precision(mpf_b, b_num, 256);

        std::cout << "\nConverted HighPrecisionNumber representations:" << std::endl;
        std::cout << "X: " << x_num.high_precision_to_string() << std::endl;
        std::cout << "Y: " << y_num.high_precision_to_string() << std::endl;
        std::cout << "A: " << a_num.high_precision_to_string() << std::endl;
        std::cout << "B: " << b_num.high_precision_to_string() << std::endl;
    }

    // Perform the calculation on the host using MPIR
    static constexpr int NUM_ITER = 1;
    computeNextXY_host(mpf_x_copy, mpf_y_copy, mpf_a, mpf_b, NUM_ITER);

    // Print the MPIR result
    std::cout << "\nHost CPU MPIR result after " << NUM_ITER << " iterations:" << std::endl;
    std::cout << "X: " << mpf_tostring(mpf_x_copy) << std::endl;
    std::cout << "Y: " << mpf_tostring(mpf_y_copy) << std::endl;

    // Convert initial mpf_t variables to HighPrecisionNumber representations
    HighPrecisionNumber a_num{};
    HighPrecisionNumber b_num{};

    mpf_to_high_precision(mpf_a, a_num, 256);
    mpf_to_high_precision(mpf_b, b_num, 256);

    HighPrecisionNumber x_num{ a_num.Size };
    HighPrecisionNumber y_num{ a_num.Size };

    std::cout << "\nConverted HighPrecisionNumber representations:" << std::endl;
    std::cout << "X: " << x_num.high_precision_to_string() << std::endl;
    std::cout << "Y: " << y_num.high_precision_to_string() << std::endl;
    std::cout << "A: " << a_num.high_precision_to_string() << std::endl;
    std::cout << "B: " << b_num.high_precision_to_string() << std::endl;

    // Assert all four are the same length
    if (x_num.Size != y_num.Size || x_num.Size != a_num.Size || x_num.Size != b_num.Size) {
        std::cerr << "Error: HighPrecisionNumber sizes do not match" << std::endl;
        return 1;
    }

    // Perform the calculation on the GPU
    HighPrecisionNumber x_gpu{ HighPrecisionNumber::CopyType::HostToDevice, x_num };
    HighPrecisionNumber y_gpu{ HighPrecisionNumber::CopyType::HostToDevice, y_num };
    HighPrecisionNumber a_gpu{ HighPrecisionNumber::CopyType::HostToDevice, a_num };
    HighPrecisionNumber b_gpu{ HighPrecisionNumber::CopyType::HostToDevice, b_num };

    // Perform the computation on the GPU
    computeNextXY_gpu(x_gpu, y_gpu, a_gpu, b_gpu, NUM_ITER);

    // After computation, copy the results back to host
    HighPrecisionNumber x_result{ HighPrecisionNumber::CopyType::DeviceToHost, x_gpu };
    HighPrecisionNumber y_result{ HighPrecisionNumber::CopyType::DeviceToHost, y_gpu };

    // Convert the HighPrecisionNumber results to strings
    std::string x_gpu_str = x_result.high_precision_to_string();
    std::string y_gpu_str = y_result.high_precision_to_string();

    // Print the GPU results
    std::cout << "\nGPU result after " << NUM_ITER << " iterations:" << std::endl;
    std::cout << "X: " << x_gpu_str << std::endl;
    std::cout << "Y: " << y_gpu_str << std::endl;

    // Convert the HighPrecisionNumber results to mpf_t for comparison
    mpf_t mpf_x_gpu_result, mpf_y_gpu_result;
    mpf_init(mpf_x_gpu_result);
    mpf_init(mpf_y_gpu_result);

    high_precision_to_mpf(x_result, mpf_x_gpu_result);
    high_precision_to_mpf(y_result, mpf_y_gpu_result);

    // Compute the differences between host and GPU results
    mpf_t mpf_x_diff, mpf_y_diff;
    mpf_init(mpf_x_diff);
    mpf_init(mpf_y_diff);

    mpf_sub(mpf_x_diff, mpf_x_copy, mpf_x_gpu_result);
    mpf_sub(mpf_y_diff, mpf_y_copy, mpf_y_gpu_result);

    // Converted GPU result
    std::cout << "\nConverted GPU result:" << std::endl;
    std::cout << "X: " << mpf_tostring(mpf_x_gpu_result) << std::endl;
    std::cout << "Y: " << mpf_tostring(mpf_y_gpu_result) << std::endl;

    // Print the differences
    std::cout << "\nDifference between host and GPU results:" << std::endl;
    std::cout << "X difference: " << mpf_tostring(mpf_x_diff) << std::endl;
    std::cout << "Y difference: " << mpf_tostring(mpf_y_diff) << std::endl;

    // Clean up MPIR variables
    mpf_clear(mpf_x);
    mpf_clear(mpf_y);
    mpf_clear(mpf_a);
    mpf_clear(mpf_b);
    mpf_clear(mpf_x_copy);
    mpf_clear(mpf_y_copy);
    mpf_clear(mpf_x_gpu_result);
    mpf_clear(mpf_y_gpu_result);
    mpf_clear(mpf_x_diff);
    mpf_clear(mpf_y_diff);

    return 0;
}
