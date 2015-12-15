#pragma once

#include <ctime>

namespace modlm {

class Timer {

public:

    Timer() { GetCurrentUTCTime(start_time); }
    ~Timer() { }

    double Elapsed();

protected:

    void GetCurrentUTCTime(timespec & spec);

    timespec start_time;

};

}
