// Timer.cu

#include "Timer.cuh"

Timer::Timer() : startTime(high_resolution_clock::now()),
                 stopTime(high_resolution_clock::now()) {}

void Timer::reset() {
    stopTime = high_resolution_clock::now();
    startTime = high_resolution_clock::now();
}

void Timer::stop() {
    stopTime = high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto endTime = (stopTime > startTime) ? stopTime : high_resolution_clock::now();
    return duration<double, std::milli>(endTime - startTime).count();
}
