// ============================================================
// utils.cpp  –  Timing, data generation, printing utilities
// ============================================================

#include "utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

// ---- Timer ------------------------------------------------
void Timer::start() {
    t0_ = std::chrono::high_resolution_clock::now();
}
void Timer::stop() {
    t1_ = std::chrono::high_resolution_clock::now();
}
double Timer::elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
}

// ---- Dataset generation ----------------------------------
// Two Gaussian blobs, each with `input_dim` features.
// Class 0: mean = -1 in all dims  |  Class 1: mean = +1 in all dims
Dataset generate_dataset(int N, int input_dim, unsigned seed) {
    Dataset ds;
    ds.N = N;
    ds.input_dim = input_dim;
    ds.X.resize(N * input_dim);
    ds.y.resize(N);

    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.5f);

    for (int i = 0; i < N; ++i) {
        int label = (i % 2);            // alternate classes
        float center = (label == 0) ? -1.0f : 1.0f;
        ds.y[i] = static_cast<float>(label);
        for (int j = 0; j < input_dim; ++j) {
            ds.X[i * input_dim + j] = center + noise(rng);
        }
    }
    return ds;
}

// ---- Pretty printing -------------------------------------
void print_separator(char c, int width) {
    for (int i = 0; i < width; ++i) std::cout << c;
    std::cout << "\n";
}

void print_banner(const std::string& title) {
    int width = 60;
    print_separator('=', width);
    int pad = (width - (int)title.size()) / 2;
    std::cout << std::string(pad, ' ') << title << "\n";
    print_separator('=', width);
}

void print_result_table(
    const std::string& label_cpu, double cpu_ms,
    const std::string& label_gpu, double gpu_ms,
    double max_diff, double cpu_acc, double gpu_acc)
{
    double speedup = cpu_ms / gpu_ms;

    print_separator('=', 60);
    std::cout << std::left
              << std::setw(30) << "Metric"
              << std::setw(15) << "CPU"
              << std::setw(15) << "GPU"
              << "\n";
    print_separator('-', 60);

    std::cout << std::left
              << std::setw(30) << "Time (ms)"
              << std::setw(15) << std::fixed << std::setprecision(3) << cpu_ms
              << std::setw(15) << gpu_ms
              << "\n";

    std::cout << std::left
              << std::setw(30) << "Accuracy (%)"
              << std::setw(15) << std::fixed << std::setprecision(2) << cpu_acc
              << std::setw(15) << gpu_acc
              << "\n";

    print_separator('-', 60);
    std::cout << std::left
              << std::setw(30) << "GPU Speedup"
              << std::fixed << std::setprecision(2) << speedup << "x\n";

    std::cout << std::left
              << std::setw(30) << "Max output difference"
              << std::scientific << std::setprecision(4) << max_diff << "\n";
    print_separator('=', 60);
}
