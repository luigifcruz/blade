#include <stdio.h>

#include <blade/memory/base.hh>

using namespace Blade;

int main() {
    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        ArrayTensor<Device::CUDA, U64> c; 
        printf("a | data = %d, size = %lu\n", a.data() != nullptr, a.size());
        printf("c | data = %d, size = %lu\n", c.data() != nullptr, c.size());
        c = std::move(a);
        printf("a | data = %d, size = %lu\n", a.data() != nullptr, a.size());
        printf("c | data = %d, size = %lu\n", c.data() != nullptr, c.size());
    }

    printf("=============================\n");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        ArrayTensor<Device::CUDA, U64> c; 
        printf("a | data = %d, size = %lu\n", a.data() != nullptr, a.size());
        c = ArrayTensor<Device::CUDA, U64>({1, 2, 3, 4});
        printf("a | data = %d, size = %lu\n", a.data() != nullptr, a.size());
    }

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        ArrayTensor<Device::CUDA, U64> c; 
    }
    
    BL_INFO("SUCCESS");
    return 0;
}
