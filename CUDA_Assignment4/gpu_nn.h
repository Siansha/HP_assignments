#pragma once
// ============================================================
// gpu_nn.h  –  CUDA (parallel) 2-layer MLP forward pass
// ============================================================
// Mirrors the CPU version exactly so we can compare outputs.
//
// Parallelism strategy:
//   Kernel 1 – gpu_matmul_relu_kernel
//     Each CUDA thread computes one output neuron of Layer 1.
//     Grid  : (N, 1, 1)  –  one block per sample
//     Block : (hidden_dim, 1, 1)  –  one thread per neuron
//     Each thread does a dot product over input_dim elements.
//
//   Kernel 2 – gpu_matmul_sigmoid_kernel
//     Each CUDA thread computes the single output neuron for
//     one sample (output_dim == 1 for binary classification).
//     Grid  : (N, 1, 1)  –  one block = one sample
//     Block : (1, 1, 1)
//     Reads the already-computed hidden activations.
//
// Why is this fast?
//   For N=10 000 samples, the CPU loops sequentially over all N.
//   The GPU runs all N samples in parallel, limited only by
//   the number of available CUDA cores.

#include <cuda_runtime.h>

// ---- CUDA error-checking helper -------------------------
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error %s:%d  %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

// ---- Batched GPU forward pass ---------------------------
// Allocates device memory, launches kernels, copies result back.
// All pointer arguments are HOST pointers.
// out:  [N]  predicted probabilities (host pointer, caller allocates)
void gpu_forward_batch(
    const float* X,          // host: [N x input_dim]
    int N,
    const float* W1,         // host: [hidden_dim x input_dim]
    const float* b1,         // host: [hidden_dim]
    const float* W2,         // host: [output_dim x hidden_dim]
    const float* b2,         // host: [output_dim]
    int input_dim,
    int hidden_dim,
    int output_dim,
    float* out);             // host: [N]  (output, pre-allocated)
