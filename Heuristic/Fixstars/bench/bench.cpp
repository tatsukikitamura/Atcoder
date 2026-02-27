/*
 * Benchmarking harness for solve().
 *
 * Usage:
 *   ./build_bench/bench-solver <input_file> [iterations=100]
 *   ./build_bench/bench-solver data/in-large-104.txt 200
 *
 * Runs solve() in a tight loop, measures each call with clock_gettime(CLOCK_MONOTONIC),
 * and prints min / median / avg / max in microseconds.
 *
 * Notes:
 *   - No fork() overhead (unlike the official main.cpp).
 *   - Cache is NOT flushed between iterations, so results are cache-warm.
 *     Add -DCOLD_CACHE to cflags to flush between iterations (slower but realistic).
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <time.h>
#include <vector>

#include "io_struct.hpp"

// solve() is defined in solve.cpp (linked via libsolve)
void solve(io_struct::input_t &input, io_struct::output_t &output);
#ifdef SOLVE_TIMING
void solve_print_timing();
#endif

// ---- Precise timer ----------------------------------------------------
static inline long long now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// ---- Optional cache flush between iterations -------------------------
#if defined(__x86_64__) && defined(COLD_CACHE)
#include <cpuid.h>
static void flush_obj(void *p, size_t n) {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    size_t cls = ((ebx >> 8) & 0xFF) * 8;
    if (cls == 0) cls = 64;
    for (size_t i = 0; i < n; i += cls)
        __builtin_ia32_clflush((char *)p + i);
    __builtin_ia32_clflush((char *)p + n - 1);
}
#else
static void flush_obj(void *, size_t) {}
#endif

// ---- Statistics -------------------------------------------------------
static void print_stats(std::vector<long long> &samples, const std::string &label) {
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double avg = (double)std::accumulate(samples.begin(), samples.end(), 0LL) / n;
    long long med = samples[n / 2];
    long long mn  = samples.front();
    long long mx  = samples.back();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[" << label << "]  n=" << n
              << "  min=" << mn / 1000.0 << "us"
              << "  median=" << med / 1000.0 << "us"
              << "  avg=" << avg / 1000.0 << "us"
              << "  max=" << mx / 1000.0 << "us"
              << "\n";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [iterations=100]\n";
        return 1;
    }
    const std::string infile = argv[1];
    const int iters = (argc >= 3) ? std::stoi(argv[2]) : 100;

    io_struct::input_t input;
    {
        std::ifstream ifs(infile);
        if (!ifs) { std::cerr << "Cannot open: " << infile << "\n"; return 1; }
        if (!io_struct::LoadInput(ifs, input)) return 1;
    }

    io_struct::output_t output;

    // Warmup (1 run, not recorded)
    io_struct::InitOutput(output);
    solve(input, output);

    std::vector<long long> samples;
    samples.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        flush_obj(&input, sizeof(input));
        flush_obj(&output, sizeof(output));
        io_struct::InitOutput(output);

        long long t0 = now_ns();
        solve(input, output);
        long long t1 = now_ns();

        samples.push_back(t1 - t0);
    }

    // Print summary
    print_stats(samples, infile);

    // Print what the last run found
    std::cout << "  result: K=" << output.K_size << "  T=" << output.T << "\n";

#ifdef SOLVE_TIMING
    // One dedicated run (cache-warm) to show per-section breakdown
    {
        io_struct::InitOutput(output);
        solve(input, output);
        solve_print_timing();
    }
#endif

    return 0;
}
