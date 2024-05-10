// Timer.cuh

#ifndef CUDA_TIMER_CUH
#define CUDA_TIMER_CUH

#include <chrono>

using namespace std::chrono;

class Timer {
public:
    Timer();
    void reset();
    void stop();
    double elapsed() const;
private:
    time_point<high_resolution_clock> startTime;
    time_point<high_resolution_clock> stopTime;
};

#endif // CUDA_TIMER_CUH
