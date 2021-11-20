#include <iostream>
#include <span>
#include <vector>

template<class T>
class CustomVector {
 public:
    CustomVector(const CustomVector&) = delete;
    CustomVector(CustomVector&&) = delete;

    CustomVector& operator=(CustomVector&& other) {
        printf("mov op\n");
        return *this;
    }

    CustomVector() : owner(false) {}

    constexpr T* data() const noexcept {
        return container.data();
    } constexpr std::size_t size() const noexcept {
        return container.size();
    }

    constexpr std::size_t size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return container.empty();
    }

    constexpr T& operator[](std::size_t idx) const {
        return container[idx];
    }

 protected:
    bool owner;
    std::span<T> container;
};

template<class T>
class HostVector : public CustomVector<T> {
 public:
    using CustomVector<T>::CustomVector;
    using CustomVector<T>::operator=;

    explicit HostVector(const std::size_t& size) {
        auto size_bytes = size * sizeof(T);
        auto ptr = static_cast<T*>(malloc(size_bytes));
        if (ptr == nullptr) {
            std::cout << "failed to allocate host memory" << std::endl;
        }
        this->container = std::span<T>(ptr, size);
        this->owner = true;
    }

    ~HostVector() {
        printf("destroy\n");
        if (this->container.data() != nullptr) {
            free(this->container.data());
        }
    }
};

void func(const HostVector<float>& olar) {
    std::cout << "func: " << olar.size() << std::endl;
}

void func2(std::span<float>& olar) {
    std::cout << "func2: " << olar.size() << std::endl;
}

int main() {
    HostVector<float> a;
    HostVector<float> b;

    a = std::move(b);
}
