#include <stdio.h>
#include <assert.h>

#include <gtest/gtest.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include "blade/memory/types.hh"
#include "blade/memory/ops.hh"

using namespace Blade;

//
// Ops Host Side Tests
//

template<typename T>
class HostComplexOpsTest : public testing::Test {};

using TypesToTest = testing::Types<F32, F64>;

TYPED_TEST_SUITE(HostComplexOpsTest, TypesToTest);

TYPED_TEST(HostComplexOpsTest, Addition) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(3.0, 4.0);
    ops::complex<TypeParam> c = a + b;
    ASSERT_EQ(c.real(), 4.0f);
    ASSERT_EQ(c.imag(), 6.0f);
}

TYPED_TEST(HostComplexOpsTest, Subtraction) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(3.0, 4.0);
    ops::complex<TypeParam> c = a - b;
    ASSERT_EQ(c.real(), -2.0f);
    ASSERT_EQ(c.imag(), -2.0f);
}

TYPED_TEST(HostComplexOpsTest, Multiplication) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(3.0, 4.0);
    ops::complex<TypeParam> c = a * b;
    ASSERT_EQ(c.real(), -5.0f);
    ASSERT_EQ(c.imag(), 10.0f);
}

TYPED_TEST(HostComplexOpsTest, Division) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(3.0, 4.0);
    ops::complex<TypeParam> c = a / b;
    ASSERT_FLOAT_EQ(c.real(), 0.44f);
    ASSERT_FLOAT_EQ(c.imag(), 0.08f);
}

TYPED_TEST(HostComplexOpsTest, Equality) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(1.0, 2.0);
    ops::complex<TypeParam> c(3.0, 4.0);
    ASSERT_EQ(a, b);
    ASSERT_NE(a, c);
}

TYPED_TEST(HostComplexOpsTest, Comparison) {
    ops::complex<TypeParam> a(1.0, 2.0);
    ops::complex<TypeParam> b(3.0, 4.0);
    ops::complex<TypeParam> c(3.0, 5.0);
    ASSERT_LT(a, b);
    ASSERT_LT(a, c);
    ASSERT_GT(b, a);
    ASSERT_GT(c, a);
}

//
// Host cuComplex vs Ops Tests
//

void PrintTo(const cuComplex& value, ::std::ostream* os) {
    *os << "(" << value.x << "," << value.y << ")";
}

TEST(cuComplexVsOpsTest, Addition) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(3.0, 4.0);
    ops::complex<F32> c = a + b;

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = cuCaddf(cu_a, cu_b);

    ASSERT_EQ(c.real(), cu_c.x);
    ASSERT_EQ(c.imag(), cu_c.y);
}

TEST(cuComplexVsOpsTest, Subtraction) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(3.0, 4.0);
    ops::complex<F32> c = a - b;

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = cuCsubf(cu_a, cu_b);

    ASSERT_EQ(c.real(), cu_c.x);
    ASSERT_EQ(c.imag(), cu_c.y);
}

TEST(cuComplexVsOpsTest, Multiplication) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(3.0, 4.0);
    ops::complex<F32> c = a * b;

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = cuCmulf(cu_a, cu_b);

    ASSERT_EQ(c.real(), cu_c.x);
    ASSERT_EQ(c.imag(), cu_c.y);
}

TEST(cuComplexVsOpsTest, Division) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(3.0, 4.0);
    ops::complex<F32> c = a / b;

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = cuCdivf(cu_a, cu_b);

    ASSERT_FLOAT_EQ(c.real(), cu_c.x);
    ASSERT_FLOAT_EQ(c.imag(), cu_c.y);
}

TEST(cuComplexVsOpsTest, Equality) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(1.0, 2.0);
    ops::complex<F32> c(3.0, 4.0);

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = make_cuComplex(c.real(), c.imag());

    ASSERT_EQ(a, cu_a);
    ASSERT_EQ(a, cu_b);
    ASSERT_NE(a, cu_c);
}

TEST(cuComplexVsOpsTest, Comparison) {
    ops::complex<F32> a(1.0, 2.0);
    ops::complex<F32> b(3.0, 4.0);
    ops::complex<F32> c(3.0, 5.0);

    cuComplex cu_a = make_cuComplex(a.real(), a.imag());
    cuComplex cu_b = make_cuComplex(b.real(), b.imag());
    cuComplex cu_c = make_cuComplex(c.real(), c.imag());

    ASSERT_LT(a, cu_b);
    ASSERT_LT(a, cu_c);
    ASSERT_GT(b, cu_a);
    ASSERT_GT(c, cu_a);
}

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
// Ops Device Side Test
//

#define EPSILON 1e-2f

__device__ bool assert_eq(half a, float b) {
    if (fabsf(__half2float(a) - b) > EPSILON) {
        printf("Assertion failed: %f != %f\n", __half2float(a), b);
        return true;
    }
    return false;
}

__device__ __managed__ bool err;

__global__ void kadd() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
    ops::complex<F16> c = a + b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCaddf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpAdd) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kadd<<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

__global__ void ksub() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
    ops::complex<F16> c = a - b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCsubf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpSub) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    ksub<<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

__global__ void kmul() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
    ops::complex<F16> c = a * b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCmulf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpMul) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kmul<<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

__global__ void kdiv() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
    ops::complex<F16> c = a / b;

    cuComplex cu_a = make_cuComplex(1.0, 2.0);
    cuComplex cu_b = make_cuComplex(3.0, 4.0);
    cuComplex cu_c = cuCdivf(cu_a, cu_b);

    err |= assert_eq(c.real(), cu_c.x);
    err |= assert_eq(c.imag(), cu_c.y);
}

TEST(HalfComplexTest, OpDiv) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kdiv<<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

__global__ void keq() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
 
    err |=  (a == b);
    err |= !(a == a);
}

TEST(HalfComplexTest, OpEq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    keq<<<1, 1>>>();
    cudaDeviceSynchronize();
    EXPECT_FALSE(err);
}

__global__ void kieq() {
    ops::complex<F16> a(1.0, 2.0);
    ops::complex<F16> b(3.0, 4.0);
 
    err |= !(a != b);
    err |=  (a != a);
}

TEST(HalfComplexTest, OpIeq) {
    ::testing::GTEST_FLAG(print_time) = true;
    err = false;
    kieq<<<1, 1>>>();
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
