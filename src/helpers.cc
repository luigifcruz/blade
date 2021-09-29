#include "bl-beamformer/helpers.hh"

namespace BL::Helpers {

Result LoadFromFile(const char* filename, void* cudaMemory, size_t size, size_t len) {
    void* host;
    FILE* file;

    if ((file = fopen(filename, "rb")) == NULL) {
        BL_FATAL("Can't open ({}) file.", filename);
        return Result::ERROR;
    }

    CUDA_CHECK(cudaMallocHost(&host, len * size), [&]{
        BL_FATAL("Can't allocate host buffer.");
    });

    fread(host, size, len, file);

    CUDA_CHECK(cudaMemcpy(cudaMemory, host, len * size, cudaMemcpyHostToDevice), [&]{
        BL_FATAL("Can't allocate device buffer.");
    });
    
    CUDA_CHECK(cudaFreeHost(host), [&]{
        BL_FATAL("Can't free host buffer.");
    });

    fclose(file);
    return Result::SUCCESS;
}

} // namespace BL::Helper
