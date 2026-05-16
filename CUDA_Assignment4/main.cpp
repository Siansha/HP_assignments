// ============================================================
// main.cpp  –  Benchmark CPU vs GPU MLP forward pass
// ============================================================
// Network: 2-layer MLP for binary classification
//   Input (128) → Hidden (256) ReLU → Output (1) Sigmoid
//
// 1. Generate a synthetic 2-class dataset
// 2. Initialise random weights (shared by both CPU and GPU)
// 3. GPU warm-up run (excludes driver init from timing)
// 4. Benchmark CPU forward pass (N_RUNS iterations, averaged)
// 5. Benchmark GPU forward pass (N_RUNS iterations, averaged)
// 6. Correctness check: max absolute difference CPU vs GPU
// 7. Print neat comparison table
// ============================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>

#include "utils.h"
#include "cpu_nn.h"
#include "gpu_nn.h"

// ============================================================
// Hyper-parameters  –  change these to stress-test
// ============================================================
static const int N          = 10000;  // number of samples
static const int INPUT_DIM  = 128;    // input features
static const int HIDDEN_DIM = 256;    // hidden neurons
static const int OUTPUT_DIM = 1;      // binary classification
static const int N_RUNS     = 20;     // benchmark repetitions

// ============================================================
// Weight initialisation (Xavier / Glorot uniform)
// ============================================================
static void init_weights(std::vector<float>& W, int fan_in, int fan_out,
                         std::mt19937& rng) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& v : W) v = dist(rng);
}

static void init_zeros(std::vector<float>& b) {
    std::fill(b.begin(), b.end(), 0.0f);
}

// ============================================================
// Main
// ============================================================
int main() {
    print_banner("CUDA Neural Network Benchmark");

    // ---- 1. Generate dataset -------------------------------
    std::cout << "\n[1/5] Generating synthetic dataset...\n";
    Dataset ds = generate_dataset(N, INPUT_DIM, /*seed=*/42);
    std::cout << "      " << N << " samples, "
              << INPUT_DIM << " features, 2 classes\n";

    // ---- 2. Initialise weights (same for CPU & GPU) --------
    std::cout << "[2/5] Initialising network weights...\n";
    std::mt19937 rng(99);

    std::vector<float> W1(HIDDEN_DIM * INPUT_DIM);
    std::vector<float> b1(HIDDEN_DIM);
    std::vector<float> W2(OUTPUT_DIM * HIDDEN_DIM);
    std::vector<float> b2(OUTPUT_DIM);

    init_weights(W1, INPUT_DIM,  HIDDEN_DIM, rng);
    init_zeros(b1);
    init_weights(W2, HIDDEN_DIM, OUTPUT_DIM, rng);
    init_zeros(b2);

    std::cout << "      W1: [" << HIDDEN_DIM << " x " << INPUT_DIM  << "]  "
              << "W2: [" << OUTPUT_DIM << " x " << HIDDEN_DIM << "]\n";

    // ---- 3. GPU warm-up ------------------------------------
    std::cout << "[3/5] Warming up GPU (one dummy forward pass)...\n";
    {
        std::vector<float> warmup_out(N);
        gpu_forward_batch(
            ds.X.data(), N,
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
            warmup_out.data());
    }
    std::cout << "      Done.\n";

    // ---- 4. CPU benchmark ----------------------------------
    std::cout << "[4/5] Running CPU benchmark (" << N_RUNS << " runs)...\n";
    std::vector<float> cpu_out(N, 0.0f);
    Timer cpu_timer;
    double cpu_total_ms = 0.0;

    for (int r = 0; r < N_RUNS; ++r) {
        cpu_timer.start();
        cpu_forward_batch(
            ds.X.data(), N,
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
            cpu_out.data());
        cpu_timer.stop();
        cpu_total_ms += cpu_timer.elapsed_ms();
    }
    double cpu_avg_ms = cpu_total_ms / N_RUNS;
    std::cout << "      Avg CPU time: " << cpu_avg_ms << " ms\n";

    // ---- 5. GPU benchmark ----------------------------------
    std::cout << "[5/5] Running GPU benchmark (" << N_RUNS << " runs)...\n";
    std::vector<float> gpu_out(N, 0.0f);
    Timer gpu_timer;
    double gpu_total_ms = 0.0;

    for (int r = 0; r < N_RUNS; ++r) {
        gpu_timer.start();
        gpu_forward_batch(
            ds.X.data(), N,
            W1.data(), b1.data(),
            W2.data(), b2.data(),
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
            gpu_out.data());
        gpu_timer.stop();
        gpu_total_ms += gpu_timer.elapsed_ms();
    }
    double gpu_avg_ms = gpu_total_ms / N_RUNS;
    std::cout << "      Avg GPU time: " << gpu_avg_ms << " ms\n";

    // ---- Correctness check ---------------------------------
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(cpu_out[i] - gpu_out[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // ---- Accuracy ------------------------------------------
    float cpu_acc = cpu_accuracy(cpu_out.data(), ds.y.data(), N);
    float gpu_acc = cpu_accuracy(gpu_out.data(), ds.y.data(), N);

    float cpu_loss = cpu_bce_loss(cpu_out.data(), ds.y.data(), N);
    float gpu_loss = cpu_bce_loss(gpu_out.data(), ds.y.data(), N);

    // ---- Print results table -------------------------------
    std::cout << "\n";
    print_result_table(
        "CPU", cpu_avg_ms,
        "GPU", gpu_avg_ms,
        max_diff, cpu_acc, gpu_acc);

    std::cout << "\nBCE Loss  →  CPU: " << cpu_loss
              << "  |  GPU: " << gpu_loss << "\n";

    // ---- Verdict -------------------------------------------
    std::cout << "\n";
    if (max_diff < 1e-4f) {
        std::cout << "[PASS] CPU and GPU outputs match (max diff < 1e-4)\n";
    } else {
        std::cout << "[WARN] Larger numerical difference detected: "
                  << max_diff << "\n";
    }

    double speedup = cpu_avg_ms / gpu_avg_ms;
    std::cout << "[INFO] GPU is " << speedup << "x faster than CPU\n";

    // ---- Architecture summary ------------------------------
    print_separator('-', 60);
    std::cout << "Network : Input(" << INPUT_DIM << ") -> "
              << "Hidden(" << HIDDEN_DIM << ", ReLU) -> "
              << "Output(" << OUTPUT_DIM << ", Sigmoid)\n";
    std::cout << "Dataset : N=" << N << " samples\n";
    std::cout << "CUDA    : grid=(" << N << "), block=(" << HIDDEN_DIM << ")\n";
    std::cout << "          Each block = 1 sample, each thread = 1 neuron\n";
    print_separator('=', 60);

    return 0;
}
