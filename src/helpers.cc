#include "bl-beamformer/helpers.hh"

namespace BL {

Result Helpers::LoadFromFile(const char* filename, void* cudaMemory, std::size_t size, std::size_t len) {
    void* host;
    FILE* file;

    BL_DEBUG("Opening file: {}", filename);

    if ((file = fopen(filename, "rb")) == NULL) {
        BL_FATAL("Can't open ({}) file.", filename);
        return Result::ERROR;
    }

    BL_CUDA_CHECK(cudaMallocHost(&host, len * size), [&]{
        BL_FATAL("Can't allocate host buffer.");
    });

    fread(host, size, len, file);

    BL_CUDA_CHECK(cudaMemcpy(cudaMemory, host, len * size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't allocate device buffer.");
    });

    BL_CUDA_CHECK(cudaFreeHost(host), [&]{
        BL_FATAL("Can't free host buffer.");
    });

    fclose(file);
    return Result::SUCCESS;
}

} // namespace BL::Helper
