#include "blade/memory/base.hh"
#include <type_traits>

using namespace Blade;

// Input Shape:       [A, F, T, P]
// Blocks per Grid:   [A, F / BLOCK_SIZE]
// Threads per Block: [BLOCK_SIZE]
//
// EXAMPLE (BLOCK_SIZE = 4):
// Input Shape:       [20, 200, 4, 2]
// Blocks per Grid:   [20, 4]
// Threads Per Block: [50]

template<typename IT,
         typename OT,
         typename XT,
         U64 A,
         U64 C,
         U64 T,
         U64 P,
         U64 BLOCK_SIZE_X,
         U64 BLOCK_SIZE_Y,
         U64 CONJUGATE_ANTENNA,
         bool USE_SHARED_MEMORY>
__global__ void correlator(const ArrayTensor<Device::CUDA, IT> input,
                                 ArrayTensor<Device::CUDA, OT> output) {
    // 1. Load antenna A and B data.
    // 2. Create temporary variables to accumulate the result.
    // 3. Add the multiply conjugate (XX = AX * CONJ(BX)) result to the temporary variables.
    // 4. Store the result in the output tensor.

    // Get Block index.

    const U64 BIX = blockIdx.x;  // Block Index X
    const U64 BIY = blockIdx.y;  // Block Index Y

    // Get Thread index.

    const U64 TIX = threadIdx.x;  // Thread Index X
    const U64 TIY = threadIdx.y;  // Thread Index Y

    // Calculate constants.

    const U64 OUTPUT_POLS = 4;                          // XX, XY, YX, YY
    const U64 AAI = BIX;                                // Antenna A Index
    const U64 CI = TIX + (BIY * BLOCK_SIZE_X);          // Channel Index
    constexpr U64 TIME_CHUNK_SIZE = T / BLOCK_SIZE_Y;
    const U64 TIME_INDEX_OFFSET = TIY * TIME_CHUNK_SIZE;

    // Cache reference antenna to shared memory.

    extern __shared__ IT memory[];
    IT (*reference)[P] = (IT (*)[P])memory;

    if constexpr (USE_SHARED_MEMORY) {
        for (U64 TI = 0; TI < TIME_CHUNK_SIZE; TI++) {
            const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
            reference[TI + TIME_INDEX_OFFSET][0] = input[ANTENNA_A_INDEX + 0];
            reference[TI + TIME_INDEX_OFFSET][1] = input[ANTENNA_A_INDEX + 1];
        }
        __syncthreads();
    }

    // Run the correlation and store the result in the output tensor.

    for (U64 ABI = AAI; ABI < A; ABI++) {
        const U64 BASELINE_INDEX = ((AAI * (2 * A - AAI + 1)) / 2) + (ABI - AAI);

        OT sumXX = OT(0.0f, 0.0f);
        OT sumXY = OT(0.0f, 0.0f);
        OT sumYX = OT(0.0f, 0.0f);
        OT sumYY = OT(0.0f, 0.0f);

        for (U64 TI = 0; TI < TIME_CHUNK_SIZE; TI++) {
            XT AVAX, AVAY = XT{};
            XT AVBX, AVBY = XT{};

            if constexpr (USE_SHARED_MEMORY) {
                AVAX = static_cast<XT>(reference[TI + TIME_INDEX_OFFSET][0]);  // Antenna Voltage A Pol X
                AVAY = static_cast<XT>(reference[TI + TIME_INDEX_OFFSET][1]);  // Antenna Voltage A Pol Y
            } else {
                const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
                AVAX = static_cast<XT>(input[ANTENNA_A_INDEX + 0]);  // Antenna Voltage A Pol X
                AVAY = static_cast<XT>(input[ANTENNA_A_INDEX + 1]);  // Antenna Voltage A Pol Y
            }

            const U64 ANTENNA_B_INDEX = (ABI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
            AVBX = static_cast<XT>(input[ANTENNA_B_INDEX + 0]);  // Antenna Voltage B Pol X
            AVBY = static_cast<XT>(input[ANTENNA_B_INDEX + 1]);  // Antenna Voltage B Pol Y

            if constexpr (CONJUGATE_ANTENNA == 1) {
                sumXX += static_cast<OT>(AVAX ^ AVBX);  // AxBx'
                sumXY += static_cast<OT>(AVAX ^ AVBY);  // AxBy'
                sumYX += static_cast<OT>(AVAY ^ AVBX);  // AyBx'
                sumYY += static_cast<OT>(AVAY ^ AVBY);  // AyBy'
            } else {
                sumXX += static_cast<OT>(AVBX ^ AVAX);  // Ax'Bx
                sumXY += static_cast<OT>(AVBY ^ AVAX);  // Ax'By
                sumYX += static_cast<OT>(AVBX ^ AVAY);  // Ay'Bx
                sumYY += static_cast<OT>(AVBY ^ AVAY);  // Ay'By
            }
        }

        const U64 OUTPUT_INDEX = (BASELINE_INDEX * C * OUTPUT_POLS) + (CI * OUTPUT_POLS);

        output[OUTPUT_INDEX + 0].atomic_add(sumXX);
        output[OUTPUT_INDEX + 1].atomic_add(sumXY);
        output[OUTPUT_INDEX + 2].atomic_add(sumYX);
        output[OUTPUT_INDEX + 3].atomic_add(sumYY);
    }
}
