#include <Eigen/Core>
#include "so3_math.hpp"
#include "structure.hpp"

#include <chrono>

int main() {
    using namespace std::chrono;
    time_point<system_clock, nanoseconds> now = system_clock::now();
    time_point<system_clock, seconds> now_sec = time_point_cast<seconds>(now);
    time_point<system_clock, nanoseconds> now_trunc = time_point_cast<nanoseconds>(std::chrono::floor<seconds>(now));
    milliseconds d = duration_cast<milliseconds>(now - now_trunc);
    
    std::cout << d << std::endl;
}

