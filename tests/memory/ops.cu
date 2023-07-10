#include <stdio.h>
#include <assert.h>

#include <gtest/gtest.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#define BL_OPS_HOST_SIDE_KEY
#include "blade/memory/device/ops.hh"
#include "blade/memory/base.hh"

using namespace Blade;

//
// Ops Alignment Test
//

TEST(ComplexSizeTest, Half) {
    EXPECT_EQ(sizeof(half2), sizeof(ops::complex<F16>));
}

TEST(ComplexSizeTest, Float) {
    EXPECT_EQ(sizeof(cuFloatComplex), sizeof(ops::complex<F32>));
}

TEST(ComplexSizeTest, Double) {
    EXPECT_EQ(sizeof(cuDoubleComplex), sizeof(ops::complex<F64>));
}

// 
// Ops Test
//

#define EPSILON 1e-2f

__device__ bool assert_eq(half a, float b) {
    if (fabsf(__half2float(a) - b) > EPSILON) {
        printf("Assertion failed: %f != %f\n", __half2float(a), b);
        return true;
    }
    return false;
}

__device__ bool assert_eq(float a, float b) {
    if (fabsf(a - b) > EPSILON) {
        printf("Assertion failed: %f != %f\n", __half2float(a), b);
        return true;
    }
    return false;
}

__device__ __managed__ bool err;

template<typename T>
__global__ void kadd() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
    ops::complex<T> c = a + b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCaddf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpAdd) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kadd<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpAdd) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kadd<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

template<typename T>
__global__ void ksub() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
    ops::complex<T> c = a - b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCsubf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpSub) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    ksub<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpSub) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    ksub<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

template<typename T>
__global__ void kmul() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
    ops::complex<T> c = a * b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCmulf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpMul) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kmul<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpMul) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kmul<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

template<typename T>
__global__ void kdiv() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
    ops::complex<T> c = a / b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCdivf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpDiv) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kdiv<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpDiv) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kdiv<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

template<typename T>
__global__ void keq() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
 
    err |=  (a == b);
    err |= !(a == a);
}

TEST(HalfComplexTest, OpEq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    keq<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpEq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    keq<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

template<typename T>
__global__ void kieq() {
    ops::complex<T> a(1.0, 2.0);
    ops::complex<T> b(3.0, 4.0);
 
    err |= !(a != b);
    err |=  (a != a);
}

TEST(HalfComplexTest, OpIeq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kieq<F16><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

TEST(SingleComplexTest, OpIeq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kieq<F32><<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

//
// Test Logic
//

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
