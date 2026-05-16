#pragma once
// ============================================================
// utils.h  –  Timing, data generation, printing utilities
// ============================================================

#include <vector>
#include <string>
#include <chrono>

// ---- Timer ------------------------------------------------
// High-resolution wall-clock timer (CPU side).
// Usage:  Timer t; t.start(); ... t.stop(); double ms = t.elapsed_ms();
class Timer {
public:
    void start();
    void stop();
    double elapsed_ms() const;          // milliseconds
private:
    std::chrono::high_resolution_clock::time_point t0_, t1_;
};

// ---- Dataset ---------------------------------------------
// Holds a simple synthetic 2-class dataset (XOR-like blobs).
struct Dataset {
    std::vector<float> X;   // [N * INPUT_DIM]  row-major
    std::vector<float> y;   // [N]  labels 0 or 1
    int N;                  // number of samples
    int input_dim;
};

// Generate a linearly-separable (two-blob) synthetic dataset.
//   N         – total samples
//   input_dim – number of features (e.g. 128)
//   seed      – RNG seed for reproducibility
Dataset generate_dataset(int N, int input_dim, unsigned seed = 42);

// ---- Pretty printing -------------------------------------
void print_separator(char c = '-', int width = 60);
void print_banner(const std::string& title);
void print_result_table(
    const std::string& label_cpu, double cpu_ms,
    const std::string& label_gpu, double gpu_ms,
    double max_diff, double cpu_acc, double gpu_acc);
