#pragma once

#include <map>
#include <set>

#include <string.h>

#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;
using std::string;

/// This timer class measures the elapsed time between two events. Timers can be
/// started and stopped repeatedly. The total time as well as the average time
/// between two events can be queried using the total() and average() methods,
/// respectively.
struct timer {

    std::chrono::system_clock::time_point begin;
    bool running;
    double   elapsed;
    unsigned counter;
    double elapsed_sq;

    timer() {
        counter = 0;
        elapsed = 0.0;
        running = false;
        elapsed_sq = 0.0;
    }

    void reset() {
        counter = 0;
        elapsed = 0.0;
        running = false;
        elapsed_sq = 0.0;
    }

    void start() {
        begin = std::chrono::high_resolution_clock::now();
        running = true;
    }

    /// The stop method returns the elapsed time since the last call of start().
    double stop() {
        if(running == false) {
            cout << "WARNING: timer::stop() has been called without calling "
                 << "timer::start() first." << endl;
            return 0.0;
        } else {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> s_double = end - begin;
            double t = s_double.count();
            counter++;
            elapsed += t;
            elapsed_sq += t*t;
            return t;
        }
    }

    double total() {
        return elapsed;
    }

    double average() {
        return elapsed/double(counter);
    }

    double deviation() {
        return sqrt(elapsed_sq/double(counter)-average()*average());
    }

    unsigned count() {
        return counter;
    }
};

namespace gt {
    std::map<string,timer> timers;

    inline bool is_master() {
        #ifdef _OPENMP
        if(omp_get_thread_num() != 0)
            return false;
        #endif

        return true;
    }
   
    inline void reset() {
        for(auto& el : timers)
            el.second.reset();
    }

    inline void print() {
        for(auto el : timers)
            cout << "gt " << el.first << ": " << el.second.total() << " s" 
                 << endl;
    }

    inline string sorted_output() {
        typedef std::pair<string,timer> pair_nt;
        auto comp = [](pair_nt a1, pair_nt a2) {
            return a1.second.total() > a2.second.total();
        };
        std::set<pair_nt, decltype(comp)> sorted(begin(timers), end(timers), comp);
        
        std::stringstream ss;
        ss.precision(4);
        ss.setf(std::ios_base::scientific);
        for(auto el : sorted) {
            timer& t = el.second;
            ss << std::setw(40) << el.first 
               << std::setw(15) << t.total()
               << std::setw(15) << t.count() 
               << std::setw(15) << t.average() 
               << std::setw(15) << t.deviation()/t.average() << endl;
        }
        return ss.str();
    }

    inline void start(string name) {
        if(is_master())
            timers[name].start();

    }

    inline void stop(string name) {
        if(is_master())
            timers[name].stop();
    }

    inline double total(string name) {
        return timers[name].total();
    }

    inline double average(string name) {
        return timers[name].average();
    }

    inline double deviation(string name) {
        return timers[name].deviation();
    }
}

