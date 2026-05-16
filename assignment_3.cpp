// =============================================================================
// Linear SVM with SGD: Sequential vs. OpenMP Parallel
// =============================================================================
// Assignment: Implement a basic SVM using OpenMP in C++
//
// HOW TO COMPILE:
//   g++ -O2 -fopenmp -o svm main.cpp
//
// HOW TO RUN:
//   ./svm
//
// OPTIONAL – control thread count at runtime:
//   OMP_NUM_THREADS=4 ./svm
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>     // Required for all OpenMP features

// =============================================================================
// CONSTANTS – tweak these to explore the behaviour
// =============================================================================
static const int    N_SAMPLES   = 5000000;  // total training points
static const int    N_FEATURES  = 2;      // dimensionality of each point (x, y)
static const int    N_EPOCHS    = 100;    // full passes over the dataset
static const double LEARNING_RATE = 0.001;
static const double LAMBDA      = 0.01;  // L2 regularisation strength (C = 1/lambda)

// =============================================================================
// DATA STRUCTURES
// =============================================================================
struct Dataset {
    // X[i] is a vector of N_FEATURES values for sample i
    std::vector<std::vector<double>> X;
    // y[i] is the label: +1 or -1
    std::vector<int> y;
    int n_samples;
    int n_features;
};

// =============================================================================
// STEP 1 – GENERATE DUMMY DATA
// =============================================================================
// Creates two linearly separable Gaussian clusters:
//   Class +1 centred at (+2, +2)
//   Class -1 centred at (-2, -2)
// A straight line through the origin cleanly separates them.
Dataset generate_data(int n_samples, int n_features, unsigned seed = 42) {
    Dataset ds;
    ds.n_samples  = n_samples;
    ds.n_features = n_features;
    ds.X.resize(n_samples, std::vector<double>(n_features));
    ds.y.resize(n_samples);

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.5); // small Gaussian noise

    for (int i = 0; i < n_samples; ++i) {
        int label = (i % 2 == 0) ? 1 : -1;  // alternate labels
        double center = (label == 1) ? 2.0 : -2.0;

        for (int j = 0; j < n_features; ++j) {
            ds.X[i][j] = center + noise(rng);
        }
        ds.y[i] = label;
    }
    return ds;
}

// =============================================================================
// HELPER – evaluate accuracy on the dataset
// =============================================================================
double evaluate(const Dataset& ds, const std::vector<double>& w, double bias) {
    int correct = 0;
    for (int i = 0; i < ds.n_samples; ++i) {
        double score = bias;
        for (int j = 0; j < ds.n_features; ++j)
            score += w[j] * ds.X[i][j];
        int pred = (score >= 0) ? 1 : -1;
        if (pred == ds.y[i]) ++correct;
    }
    return 100.0 * correct / ds.n_samples;
}

// =============================================================================
// STEP 2 – SEQUENTIAL SVM TRAINING (Hinge loss + L2 regularisation)
// =============================================================================
// The SVM optimisation objective (primal form):
//
//   min  (lambda/2)||w||^2  +  (1/N) * sum_i max(0, 1 - y_i*(w·x_i + b))
//    w,b
//
// Gradient of the hinge loss for one sample:
//   If y_i*(w·x_i + b) >= 1  →  no loss; gradient = lambda * w
//   Else                      →  gradient = lambda * w - y_i * x_i
//                                           bias grad = -y_i
//
// We accumulate gradients over all N samples, then take one gradient step.
// =============================================================================
std::vector<double> train_sequential(const Dataset& ds,
                                     double& bias_out,
                                     double& elapsed_ms) {
    int N = ds.n_samples;
    int F = ds.n_features;

    std::vector<double> w(F, 0.0);  // weight vector (initialised to 0)
    double bias = 0.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        // Gradient accumulators for this epoch
        std::vector<double> grad_w(F, 0.0);
        double grad_b = 0.0;

        // ---- Inner loop: accumulate gradients over all training samples ----
        // This is the loop we will parallelise in the OpenMP version below.
        for (int i = 0; i < N; ++i) {
            // Compute the decision value: w · x_i + b
            double decision = bias;
            for (int j = 0; j < F; ++j)
                decision += w[j] * ds.X[i][j];

            // Check the hinge condition: y_i * decision
            double margin = ds.y[i] * decision;

            if (margin < 1.0) {
                // Sample is inside the margin or misclassified → hinge loss is active
                // Gradient contribution from hinge term: -y_i * x_i
                for (int j = 0; j < F; ++j)
                    grad_w[j] += -ds.y[i] * ds.X[i][j];
                grad_b += -ds.y[i];
            }
            // If margin >= 1, hinge loss is 0; only regularisation contributes
        }

        // ---- Weight update: gradient descent step ----
        // Full gradient = regularisation term + (1/N) * hinge gradient
        for (int j = 0; j < F; ++j)
            w[j] -= LEARNING_RATE * (LAMBDA * w[j] + grad_w[j] / N);
        bias -= LEARNING_RATE * (grad_b / N);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bias_out = bias;
    return w;
}

// =============================================================================
// STEP 3 – PARALLEL SVM TRAINING (OpenMP)
// =============================================================================
// The mathematics are IDENTICAL to the sequential version.
// The only difference is that the inner gradient-accumulation loop is
// parallelised with OpenMP.
//
// KEY OPENMP CONCEPTS USED:
//
//  #pragma omp parallel for reduction(+:var)
//  ──────────────────────────────────────────
//  • Splits the loop iterations across available threads automatically.
//  • Each thread works on its own private copy of `var`.
//  • At the end of the parallel section, all private copies are ADDED (+)
//    together into the shared `var` – this avoids race conditions.
//
//  Why is reduction needed?
//  ─────────────────────────
//  Without reduction, multiple threads could read and write `grad_b`
//  simultaneously (a DATA RACE), producing wrong, non-deterministic results.
//  The reduction clause gives each thread a private accumulator and merges
//  them safely after all threads finish.
//
//  What about grad_w (a vector)?
//  ──────────────────────────────
//  OpenMP's built-in reduction doesn't support std::vector directly.
//  Instead we use a thread-private partial array and an atomic add, OR
//  (simpler & equally correct) a #pragma omp critical section to protect
//  the update. Here we use a local per-thread array + a final critical
//  reduction, which is efficient for small F.
// =============================================================================
std::vector<double> train_parallel(const Dataset& ds,
                                   double& bias_out,
                                   double& elapsed_ms) {
    int N = ds.n_samples;
    int F = ds.n_features;

    std::vector<double> w(F, 0.0);
    double bias = 0.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        // Shared gradient accumulators (written by the reduction below)
        std::vector<double> grad_w(F, 0.0);
        double grad_b = 0.0;

        // ── OpenMP parallel region ────────────────────────────────────────
        // #pragma omp parallel: spawns a team of threads.
        // Each thread executes the block that follows.
        #pragma omp parallel
        {
            // Thread-local (private) accumulators.
            // Each thread accumulates into its own copy → no race condition.
            std::vector<double> local_grad_w(F, 0.0);
            double local_grad_b = 0.0;

            // #pragma omp for: distributes loop iterations across threads.
            // The schedule(static) clause splits iterations into equal-sized
            // contiguous chunks, one per thread. This is predictable and has
            // low overhead – good when all iterations take similar time.
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                double decision = bias;  // bias is read-only here → no race
                for (int j = 0; j < F; ++j)
                    decision += w[j] * ds.X[i][j];

                double margin = ds.y[i] * decision;

                if (margin < 1.0) {
                    for (int j = 0; j < F; ++j)
                        local_grad_w[j] += -ds.y[i] * ds.X[i][j];
                    local_grad_b += -ds.y[i];
                }
            }
            // ── End of distributed loop ───────────────────────────────────

            // Now merge each thread's local accumulator into the shared one.
            // #pragma omp critical: only ONE thread enters this block at a time.
            // This is safe because it happens only once per thread per epoch
            // (not inside the hot loop), so the serialisation cost is tiny.
            #pragma omp critical
            {
                for (int j = 0; j < F; ++j)
                    grad_w[j] += local_grad_w[j];
                grad_b += local_grad_b;
            }
        }
        // ── End of parallel region ────────────────────────────────────────
        // All threads have joined back here. grad_w and grad_b now hold the
        // same totals as in the sequential version.

        // Weight update – sequential, same formula as before
        for (int j = 0; j < F; ++j)
            w[j] -= LEARNING_RATE * (LAMBDA * w[j] + grad_w[j] / N);
        bias -= LEARNING_RATE * (grad_b / N);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bias_out = bias;
    return w;
}

// =============================================================================
// STEP 4 – MAIN: run both, verify, compare
// =============================================================================
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Print OpenMP thread count so the user can verify parallelism
    int n_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single   // only one thread executes this
        n_threads = omp_get_num_threads();
    }

    std::cout << "==============================================\n";
    std::cout << "  Linear SVM: Sequential vs OpenMP Parallel  \n";
    std::cout << "==============================================\n";
    std::cout << "Samples     : " << N_SAMPLES     << "\n";
    std::cout << "Features    : " << N_FEATURES    << "\n";
    std::cout << "Epochs      : " << N_EPOCHS      << "\n";
    std::cout << "LR / Lambda : " << LEARNING_RATE << " / " << LAMBDA << "\n";
    std::cout << "OMP threads : " << n_threads     << "\n\n";

    // --- Generate data (same for both runs) ---
    std::cout << "Generating dataset...\n";
    Dataset ds = generate_data(N_SAMPLES, N_FEATURES);

    // -------------------------------------------------------------------------
    // Sequential run
    // -------------------------------------------------------------------------
    std::cout << "\n[1/2] Training SEQUENTIAL SVM...\n";
    double seq_bias = 0.0, seq_ms = 0.0;
    std::vector<double> seq_w = train_sequential(ds, seq_bias, seq_ms);

    double seq_acc = evaluate(ds, seq_w, seq_bias);
    std::cout << "  Time     : " << seq_ms   << " ms\n";
    std::cout << "  Accuracy : " << seq_acc  << " %\n";
    std::cout << "  Weights  : w = [";
    for (int j = 0; j < N_FEATURES; ++j)
        std::cout << seq_w[j] << (j + 1 < N_FEATURES ? ", " : "");
    std::cout << "],  bias = " << seq_bias << "\n";

    // -------------------------------------------------------------------------
    // Parallel run
    // -------------------------------------------------------------------------
    std::cout << "\n[2/2] Training PARALLEL SVM (OpenMP)...\n";
    double par_bias = 0.0, par_ms = 0.0;
    std::vector<double> par_w = train_parallel(ds, par_bias, par_ms);

    double par_acc = evaluate(ds, par_w, par_bias);
    std::cout << "  Time     : " << par_ms   << " ms\n";
    std::cout << "  Accuracy : " << par_acc  << " %\n";
    std::cout << "  Weights  : w = [";
    for (int j = 0; j < N_FEATURES; ++j)
        std::cout << par_w[j] << (j + 1 < N_FEATURES ? ", " : "");
    std::cout << "],  bias = " << par_bias << "\n";

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------
    std::cout << "\n==============================================\n";
    std::cout << "  RESULTS COMPARISON\n";
    std::cout << "==============================================\n";

    // Check that weights match within floating-point tolerance
    bool weights_match = true;
    double max_diff = 0.0;
    for (int j = 0; j < N_FEATURES; ++j) {
        double diff = std::fabs(seq_w[j] - par_w[j]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-9) weights_match = false;
    }
    if (std::fabs(seq_bias - par_bias) > 1e-9) weights_match = false;

    std::cout << "  Max weight difference : " << max_diff
              << (weights_match ? "  ✓ (mathematically identical)\n"
                               : "  ✗ (WARNING: results differ!)\n");

    double speedup = seq_ms / par_ms;
    std::cout << "  Sequential time  : " << seq_ms << " ms\n";
    std::cout << "  Parallel time    : " << par_ms << " ms\n";
    std::cout << "  Speedup          : " << std::setprecision(2) << speedup << "x";
    if (speedup > 1.0)
        std::cout << "  ✓ OpenMP is faster!\n";
    else
        std::cout << "  (dataset may be too small to overcome thread-spawn overhead)\n";

    std::cout << "==============================================\n";
    return 0;
}