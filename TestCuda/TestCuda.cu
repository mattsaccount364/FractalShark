// Chat-GPT has some issues, and it's unfortunate that I can't use it for this task.
// It helpfully filled that prior comment in for me. I'm not sure what to say here.
// LOL - I'm not sure what to say here either. I'm not sure what to say here either.
// I'm not sure what to say here either. I'm not sure what to say here either.
// All work and no play makes Jack a dull a ...

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Constants for the maximum length of integer and fractional parts
#define MAX_LEN 256
#define MAX_LEN_FRAC 256
#define BASE 10000
#define NUM_ITER 1

// Struct to hold both integer and fractional parts of the high precision number
struct HighPrecisionNumber {
    std::vector<int> int_part;
    std::vector<int> frac_part;
};

// Function to convert mpf_t to HighPrecisionNumber
void mpf_to_array(const mpf_t mpf_val, HighPrecisionNumber &number, int precision) {
    // Clear the vectors
    number.int_part.clear();
    number.frac_part.clear();

    // Get the value as a string
    mp_exp_t exp;
    char *str = mpf_get_str(NULL, &exp, 10, precision + 1, mpf_val);
    std::string val_str(str);
    free(str);

    // Ensure the string is long enough to include fractional part with leading zeros
    if (val_str.length() < (size_t)exp + precision) {
        val_str.append((exp + precision) - val_str.length(), '0');
    }

    // Handle integer part
    std::string int_part_str = exp > 0 ? val_str.substr(0, exp) : "0";
    if (int_part_str.empty()) {
        int_part_str = "0";
    }
    for (int i = int_part_str.length(); i > 0; i -= 4) {
        int end = i;
        int start = std::max(0, i - 4);
        number.int_part.push_back(std::stoi(int_part_str.substr(start, end - start)));
    }
    std::reverse(number.int_part.begin(), number.int_part.end());

    // Handle fractional part
    std::string frac_part_str = exp < (int)val_str.length() ? val_str.substr(exp) : "";
    for (size_t i = 0; i < frac_part_str.length(); i += 4) {
        int start = i;
        int end = std::min((size_t)(i + 4), frac_part_str.length());
        std::string segment = frac_part_str.substr(start, end - start);
        // Ensure the segment has leading zeros
        while (segment.length() < 4) {
            segment += '0';
        }
        number.frac_part.push_back(std::stoi(segment));
    }
}

// Kernel to compute the next values for x and y in a single pass
__global__ void computeNextXY(int *x_int, int *x_frac, int *y_int, int *y_frac, int *a_int, int *a_frac, int *b_int, int *b_frac, int *x_new_int, int *x_new_frac, int *y_new_int, int *y_new_frac, int max_len, int num_iter) {
    extern __shared__ int shared_mem[];

    int *x_squared = shared_mem;
    int *y_squared = x_squared + max_len * 2 + max_len * 2;
    int *xy = y_squared + max_len * 2 + max_len * 2;
    int *two_xy = xy + max_len * 2 + max_len * 2;
    int *temp_x = two_xy + max_len * 2 + max_len * 2;

    int tx = threadIdx.x;

    for (int iter = 0; iter < num_iter; ++iter) {
        // Reset intermediate arrays
        if (tx < max_len * 2 + max_len * 2) {
            x_squared[tx] = 0;
            y_squared[tx] = 0;
            xy[tx] = 0;
            two_xy[tx] = 0;
            temp_x[tx] = 0;
        }
        __syncthreads();

        // Perform the calculations for integer part
        if (tx < max_len) {
            for (int i = 0; i < max_len; ++i) {
                atomicAdd(&x_squared[tx + i], x_int[tx] * x_int[i]);
                atomicAdd(&y_squared[tx + i], y_int[tx] * y_int[i]);
                atomicAdd(&xy[tx + i], x_int[tx] * y_int[i]);
            }
        }
        __syncthreads();

        // Perform the calculations for fractional part
        if (tx < max_len) {
            for (int i = 0; i < max_len; ++i) {
                atomicAdd(&x_squared[tx + max_len + i], x_frac[tx] * x_frac[i]);
                atomicAdd(&y_squared[tx + max_len + i], y_frac[tx] * y_frac[i]);
                atomicAdd(&xy[tx + max_len + i], x_frac[tx] * y_frac[i]);
            }
        }
        __syncthreads();

        // Propagate carry for x_squared
        for (int i = 0; i < max_len * 2 - 1; ++i) {
            int carry = x_squared[i] / BASE;
            x_squared[i] %= BASE;
            atomicAdd(&x_squared[i + 1], carry);
        }
        __syncthreads();

        // Propagate carry for y_squared
        for (int i = 0; i < max_len * 2 - 1; ++i) {
            int carry = y_squared[i] / BASE;
            y_squared[i] %= BASE;
            atomicAdd(&y_squared[i + 1], carry);
        }
        __syncthreads();

        // Propagate carry for xy
        for (int i = 0; i < max_len * 2 - 1; ++i) {
            int carry = xy[i] / BASE;
            xy[i] %= BASE;
            atomicAdd(&xy[i + 1], carry);
        }
        __syncthreads();

        // Calculate 2xy
        if (tx < max_len * 2) {
            two_xy[tx] = xy[tx] * 2;
        }
        __syncthreads();

        // Calculate x_new and y_new
        if (tx < max_len * 2) {
            temp_x[tx] = x_squared[tx] - y_squared[tx];
            if (tx < max_len) {
                x_new_int[tx] = temp_x[tx] + a_int[tx];
                y_new_int[tx] = two_xy[tx] + b_int[tx];
            } else {
                x_new_frac[tx - max_len] = temp_x[tx] + a_frac[tx - max_len];
                y_new_frac[tx - max_len] = two_xy[tx] + b_frac[tx - max_len];
            }

            // Propagate carry for x_new
            for (int i = 0; i < max_len * 2 - 1; ++i) {
                if (tx == i) {
                    int carry = (tx < max_len) ? x_new_int[i] / BASE : x_new_frac[i - max_len] / BASE;
                    if (tx < max_len) {
                        x_new_int[i] %= BASE;
                        atomicAdd(&x_new_int[i + 1], carry);
                    } else {
                        x_new_frac[i - max_len] %= BASE;
                        atomicAdd(&x_new_frac[i + 1 - max_len], carry);
                    }
                }
                __syncthreads();
            }

            // Propagate carry for y_new
            for (int i = 0; i < max_len * 2 - 1; ++i) {
                if (tx == i) {
                    int carry = (tx < max_len) ? y_new_int[i] / BASE : y_new_frac[i - max_len] / BASE;
                    if (tx < max_len) {
                        y_new_int[i] %= BASE;
                        atomicAdd(&y_new_int[i + 1], carry);
                    } else {
                        y_new_frac[i - max_len] %= BASE;
                        atomicAdd(&y_new_frac[i + 1 - max_len], carry);
                    }
                }
                __syncthreads();
            }
        }

        // Swap x and x_new, y and y_new for the next iteration
        if (tx < max_len * 2) {
            if (tx < max_len) {
                x_int[tx] = x_new_int[tx];
                y_int[tx] = y_new_int[tx];
            } else {
                x_frac[tx - max_len] = x_new_frac[tx - max_len];
                y_frac[tx - max_len] = y_new_frac[tx - max_len];
            }
        }
        __syncthreads();
    }
}

// Function to print the result in a readable format
std::string array_to_string(const HighPrecisionNumber &number) {
    std::stringstream ss;

    // Handle the integer part
    bool leading_zero = true;
    for (size_t i = 0; i < number.int_part.size(); ++i) {
        if (leading_zero) {
            ss << number.int_part[i];  // The first segment may not have leading zeros
            leading_zero = false;
        } else {
            ss << std::setw(4) << std::setfill('0') << number.int_part[i];  // Ensure 4 digits with leading zeros
        }
    }
    if (leading_zero) {
        ss << "0";  // Handle the case where the integer part is zero
    }

    // Handle the fractional part
    ss << ".";
    for (size_t i = 0; i < number.frac_part.size(); ++i) {
        ss << std::setw(4) << std::setfill('0') << number.frac_part[i];  // Ensure 4 digits with leading zeros
    }

    return ss.str();
}

// Function to convert mpf_t to string
std::string mpf_tostring(mpf_t mpf_val) {
    constexpr size_t number_of_chars = 128 * 1024;
    std::vector<char> temp(number_of_chars);
    gmp_snprintf(temp.data(), number_of_chars, "%.Fe", mpf_val);
    std::string result(temp.data());
    return result;
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
        mpf_mul(temp_y, x, y); // xy
        mpf_mul_ui(two_xy, temp_y, 2); // 2xy

        mpf_sub(temp_x, x_squared, y_squared); // x^2 - y^2
        mpf_add(temp_x, temp_x, a); // x^2 - y^2 + a
        mpf_add(temp_y, two_xy, b); // 2xy + b

        mpf_set(x, temp_x);
        mpf_set(y, temp_y);
    }

    mpf_clear(x_squared);
    mpf_clear(y_squared);
    mpf_clear(two_xy);
    mpf_clear(temp_x);
    mpf_clear(temp_y);
}

// Test function for array_to_string
void test_array_to_string() {
    // First convert a string to a HighPrecisionNumber, such as "12345.678912"
    // using mpf_tostring
    HighPrecisionNumber number;
    mpf_t mpf_val;
    mpf_init(mpf_val);
    mpf_set_str(mpf_val, "12345.678912", 10);
    mpf_to_array(mpf_val, number, 256);

    // Then convert the HighPrecisionNumber back to a string using array_to_string
    std::string result = array_to_string(number);
    std::cout << "Result: " << result << std::endl;

    // Convert the string back to a HighPrecisionNumber using mpf_to_array
    HighPrecisionNumber number2;
    mpf_to_array(mpf_val, number2, 256);

    // Convert the HighPrecisionNumber back to a string using array_to_string
    std::string result2 = array_to_string(number2);
    std::cout << "Result2: " << result2 << std::endl;

    // These should match
    if (result != result2) {
        std::cerr << "Error: array_to_string does not match mpf_tostring" << std::endl;
    }

    // Convert result2 to mpf_t:
    mpf_t mpf_val2;
    mpf_init(mpf_val2);
    mpf_set_str(mpf_val2, result2.c_str(), 10);

    // Print mpf_val2:
    std::cout << "mpf_val2: " << mpf_tostring(mpf_val2) << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <x> <y> <a> <b>" << std::endl;
        return 1;
    }

    mpf_set_default_prec(1024);  // Set precision for MPIR floating point

    //test_array_to_string();
    //return 0;

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

    // Make a copy of the input values
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

    // Perform the calculation on the host using MPIR
    computeNextXY_host(mpf_y_copy, mpf_y_copy, mpf_a, mpf_b, NUM_ITER);

    // Print the MPIR result
    std::cout << "Host CPU MPIR result:" << std::endl;
    std::cout << "X: " << mpf_tostring(mpf_y_copy) << std::endl;
    std::cout << "Y: " << mpf_tostring(mpf_y_copy) << std::endl;

    // Convert mpf_t to array representation
    HighPrecisionNumber x_num, y_num, a_num, b_num;
    mpf_to_array(mpf_x, x_num, 256);
    mpf_to_array(mpf_y, y_num, 256);
    mpf_to_array(mpf_a, a_num, 256);
    mpf_to_array(mpf_b, b_num, 256);

    // Print the array representation
    std::cout << "Array representation:" << std::endl;
    std::cout << "X: " << array_to_string(x_num) << std::endl;
    std::cout << "Y: " << array_to_string(y_num) << std::endl;
    std::cout << "A: " << array_to_string(a_num) << std::endl;
    std::cout << "B: " << array_to_string(b_num) << std::endl;

    // Allocate and initialize host arrays
    const int max_len = MAX_LEN;
    const int max_len_frac = MAX_LEN_FRAC;

    int *h_x_int = new int[max_len]();
    int *h_x_frac = new int[max_len_frac]();
    int *h_y_int = new int[max_len]();
    int *h_y_frac = new int[max_len_frac]();
    int *h_a_int = new int[max_len]();
    int *h_a_frac = new int[max_len_frac]();
    int *h_b_int = new int[max_len]();
    int *h_b_frac = new int[max_len_frac]();
    int *h_x_new_int = new int[max_len]();
    int *h_x_new_frac = new int[max_len_frac]();
    int *h_y_new_int = new int[max_len]();
    int *h_y_new_frac = new int[max_len_frac]();

    std::copy(x_num.int_part.begin(), x_num.int_part.end(), h_x_int);
    std::copy(x_num.frac_part.begin(), x_num.frac_part.end(), h_x_frac);
    std::copy(y_num.int_part.begin(), y_num.int_part.end(), h_y_int);
    std::copy(y_num.frac_part.begin(), y_num.frac_part.end(), h_y_frac);
    std::copy(a_num.int_part.begin(), a_num.int_part.end(), h_a_int);
    std::copy(a_num.frac_part.begin(), a_num.frac_part.end(), h_a_frac);
    std::copy(b_num.int_part.begin(), b_num.int_part.end(), h_b_int);
    std::copy(b_num.frac_part.begin(), b_num.frac_part.end(), h_b_frac);

    int *d_x_int, *d_x_frac, *d_y_int, *d_y_frac, *d_a_int, *d_a_frac, *d_b_int, *d_b_frac, *d_x_new_int, *d_x_new_frac, *d_y_new_int, *d_y_new_frac;

    // Allocate device memory
    cudaMalloc((void **)&d_x_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_x_frac, max_len_frac * sizeof(int));
    cudaMalloc((void **)&d_y_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_y_frac, max_len_frac * sizeof(int));
    cudaMalloc((void **)&d_a_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_a_frac, max_len_frac * sizeof(int));
    cudaMalloc((void **)&d_b_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_b_frac, max_len_frac * sizeof(int));
    cudaMalloc((void **)&d_x_new_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_x_new_frac, max_len_frac * sizeof(int));
    cudaMalloc((void **)&d_y_new_int, max_len * sizeof(int));
    cudaMalloc((void **)&d_y_new_frac, max_len_frac * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_x_int, h_x_int, max_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_frac, h_x_frac, max_len_frac * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_int, h_y_int, max_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_frac, h_y_frac, max_len_frac * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_int, h_a_int, max_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_frac, h_a_frac, max_len_frac * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_int, h_b_int, max_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_frac, h_b_frac, max_len_frac * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to compute next values of x and y for NUM_ITER iterations
    int threadsPerBlock = max_len + max_len_frac;
    int blocksPerGrid = 1;
    size_t sharedMemSize = 5 * (max_len * 2 + max_len_frac * 2) * sizeof(int);

    computeNextXY << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_x_int, d_x_frac, d_y_int, d_y_frac, d_a_int, d_a_frac, d_b_int, d_b_frac, d_x_new_int, d_x_new_frac, d_y_new_int, d_y_new_frac, max_len, NUM_ITER);

    // Copy result back to host
    cudaMemcpy(h_x_new_int, d_x_new_int, max_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_new_frac, d_x_new_frac, max_len_frac * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_new_int, d_y_new_int, max_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_new_frac, d_y_new_frac, max_len_frac * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert the results to readable format
    HighPrecisionNumber x_new_num = { std::vector<int>(h_x_new_int, h_x_new_int + max_len), std::vector<int>(h_x_new_frac, h_x_new_frac + max_len_frac) };
    HighPrecisionNumber y_new_num = { std::vector<int>(h_y_new_int, h_y_new_int + max_len), std::vector<int>(h_y_new_frac, h_y_new_frac + max_len_frac) };
    std::string x_result = array_to_string(x_new_num);
    std::string y_result = array_to_string(y_new_num);

    // Print CUDA result
    std::cout << "CUDA result:" << std::endl;
    std::cout << "X_new: " << x_result << std::endl;
    std::cout << "Y_new: " << y_result << std::endl;

    // Cleanup
    delete[] h_x_int;
    delete[] h_x_frac;
    delete[] h_y_int;
    delete[] h_y_frac;
    delete[] h_a_int;
    delete[] h_a_frac;
    delete[] h_b_int;
    delete[] h_b_frac;
    delete[] h_x_new_int;
    delete[] h_x_new_frac;
    delete[] h_y_new_int;
    delete[] h_y_new_frac;
    cudaFree(d_x_int);
    cudaFree(d_x_frac);
    cudaFree(d_y_int);
    cudaFree(d_y_frac);
    cudaFree(d_a_int);
    cudaFree(d_a_frac);
    cudaFree(d_b_int);
    cudaFree(d_b_frac);
    cudaFree(d_x_new_int);
    cudaFree(d_x_new_frac);
    cudaFree(d_y_new_int);
    cudaFree(d_y_new_frac);

    mpf_clear(mpf_x);
    mpf_clear(mpf_y);
    mpf_clear(mpf_a);
    mpf_clear(mpf_b);

    return 0;
}
