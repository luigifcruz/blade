#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <blade/memory/base.hh>

using namespace Blade;

void PrintVarDebug(const std::string& varName, const auto& a) {
    BL_DEBUG("{} | data = {}, size = {}, refs = {}, ptr = {}", 
             varName, a.data() != nullptr, a.size(), a.refs(), 
             fmt::ptr(a.data()));
}

int main() {
    {
        ArrayTensor<Device::CUDA, U64> a;
        PrintVarDebug("a", a);
        assert(a.size() == 0);
        assert(a.refs() == 0);
        assert(a.data() == nullptr);

        BL_INFO("Vector empty creation test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        BL_INFO("Vector filled creation test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        void* ptr = (void*)0xdeadbeefc00fee;
        ArrayTensor<Device::CUDA, U64> a(ptr, {1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 0);
        assert(a.data() == ptr);

        BL_INFO("Vector zombie creation test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        ArrayTensor<Device::CUDA, U64> c(a);
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);

        BL_INFO("Vector copy constructor test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        ArrayTensor<Device::CUDA, U64> c(std::move(a));
        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 1);
        assert(c.data() != nullptr);

        PrintVarDebug("a", a);
        assert(a.size() == 0);
        assert(a.refs() == 0);
        assert(a.data() == nullptr);

        BL_INFO("Vector move constructor test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        ArrayTensor<Device::CUDA, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.refs() == 0);
        assert(c.data() == nullptr);

        c = a; 

        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 2);
        assert(a.data() != nullptr);

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 2);
        assert(c.data() != nullptr);

        BL_INFO("Vector copy test successful!");
    }

    BL_INFO("---------------------------------------------");

    {
        ArrayTensor<Device::CUDA, U64> a({1, 2, 3, 4});
        PrintVarDebug("a", a);
        assert(a.size() == 24);
        assert(a.refs() == 1);
        assert(a.data() != nullptr);

        ArrayTensor<Device::CUDA, U64> c; 
        PrintVarDebug("c", c);
        assert(c.size() == 0);
        assert(c.refs() == 0);
        assert(c.data() == nullptr);

        c = std::move(a); 

        PrintVarDebug("a", a);
        assert(a.size() == 0);
        assert(a.refs() == 0);
        assert(a.data() == nullptr);

        PrintVarDebug("c", c);
        assert(c.size() == 24);
        assert(c.refs() == 1);
        assert(c.data() != nullptr);

        BL_INFO("Vector move test successful!");
    }

    return 0;
}
