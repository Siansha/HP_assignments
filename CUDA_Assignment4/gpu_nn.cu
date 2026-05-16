// ============================================================
// gpu_nn.cu  –  CUDA (parallel) 2-layer MLP forward pass
// ============================================================
//
// KEY IDEA: Every sample is processed by a separate block
// of threads in parallel, so the whole batch runs at once.
//
// Memory layout (device):
//   d_X     [N x input_dim]       – all input samples
//   d_W1    [hidden_dim x input_dim]
//   d_b1    [hidden_dim]
//   d_W2    [output_dim x hidden_dim]
//   d_b2    [output_dim]
//   d_H     [N x hidden_dim]      – hidden activations (scratch)
//   d_out   [N]                   – final probabilities
//
// ============================================================

#include "gpu_nn.h"
#include <cstdio>
#include <cmath>

// ===========================================================
// KERNEL 1: Layer 1 – linear + ReLU
// ===========================================================
// Grid  : dim3(N)          – one block per sample
// Block : dim3(hidden_dim) – one thread per hidden neuron
//
// Thread (blockIdx.x, threadIdx.x) computes hidden neuron
// threadIdx.x for sample blockIdx.x.
// ===========================================================
__global__ void gpu_matmul_relu_kernel(
    const float* __restrict__ X,   // [N x input_dim]
    const float* __restrict__ W1,  // [hidden_dim x input_dim]
    const float* __restrict__ b1,  // [hidden_dim]
    float*       __restrict__ H,   // [N x hidden_dim]  output
    int input_dim,
    int hidden_dim)
{
    int sample  = blockIdx.x;   // which sample (0..N-1)
    int neuron  = threadIdx.x;  // which hidden neuron (0..hidden_dim-1)

    // Guard: skip if out of range (for incomplete last block)
    if (neuron >= hidden_dim) return;

    // Compute dot product: W1[neuron, :] · X[sample, :]
    float acc = b1[neuron];
    const float* x_row  = X  + sample * input_dim;   // input row
    const float* w_row  = W1 + neuron * input_dim;   // weight row

    for (int j = 0; j < input_dim; ++j) {
        acc += w_row[j] * x_row[j];
    }

    // Apply ReLU and store into hidden activation matrix
    H[sample * hidden_dim + neuron] = (acc > 0.0f) ? acc : 0.0f;
}

// ===========================================================
// KERNEL 2: Layer 2 – linear + Sigmoid  (output_dim == 1)
// ===========================================================
// Grid  : dim3(N)   – one block per sample
// Block : dim3(1)   – single thread per sample (output dim = 1)
//
// Each thread reads the hidden activations of its sample
// (already stored in H) and produces one probability.
// ===========================================================
__global__ void gpu_matmul_sigmoid_kernel(
    const float* __restrict__ H,    // [N x hidden_dim]
    const float* __restrict__ W2,   // [hidden_dim]  (output_dim=1)
    const float* __restrict__ b2,   // [1]
    float*       __restrict__ out,  // [N]  output probabilities
    int hidden_dim)
{
    int sample = blockIdx.x;        // which sample

    float logit = b2[0];
    const float* h_row = H + sample * hidden_dim;

    for (int h = 0; h < hidden_dim; ++h) {
        logit += W2[h] * h_row[h];
    }

    // Sigmoid activation
    out[sample] = 1.0f / (1.0f + expf(-logit));
}

// ===========================================================
// Host function: orchestrates device memory and kernel calls
// ===========================================================
void gpu_forward_batch(
    const float* X,
    int N,
    const float* W1,
    const float* b1,
    const float* W2,
    const float* b2,
    int input_dim,
    int hidden_dim,
    int output_dim,
    float* out)
{
    // ---- Sizes in bytes ------------------------------------
    size_t sz_X   = (size_t)N * input_dim  * sizeof(float);
    size_t sz_W1  = (size_t)hidden_dim * input_dim  * sizeof(float);
    size_t sz_b1  = (size_t)hidden_dim * sizeof(float);
    size_t sz_W2  = (size_t)hidden_dim * sizeof(float); // output_dim=1
    size_t sz_b2  = (size_t)1 * sizeof(float);
    size_t sz_H   = (size_t)N * hidden_dim * sizeof(float);
    size_t sz_out = (size_t)N * sizeof(float);

    // ---- Allocate device memory ----------------------------
    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2, *d_H, *d_out;   
    CUDA_CHECK(cudaMalloc(&d_X,   sz_X));
    CUDA_CHECK(cudaMalloc(&d_W1,  sz_W1));
    CUDA_CHECK(cudaMalloc(&d_b1,  sz_b1));
    CUDA_CHECK(cudaMalloc(&d_W2,  sz_W2));
    CUDA_CHECK(cudaMalloc(&d_b2,  sz_b2));
    CUDA_CHECK(cudaMalloc(&d_H,   sz_H));
    CUDA_CHECK(cudaMalloc(&d_out, sz_out));

    // ---- Copy inputs from host to device (H2D) ------------
    CUDA_CHECK(cudaMemcpy(d_X,  X,  sz_X,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, W1, sz_W1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1, sz_b1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2, sz_W2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2, sz_b2, cudaMemcpyHostToDevice));

    // ---- Launch Kernel 1: Layer 1 (linear + ReLU) ---------
    //   Grid:  N blocks  (one per sample)
    //   Block: hidden_dim threads (one per neuron)
    dim3 grid1(N);
    dim3 block1(hidden_dim);
    gpu_matmul_relu_kernel<<<grid1, block1>>>(
        d_X, d_W1, d_b1, d_H, input_dim, hidden_dim);
    CUDA_CHECK(cudaGetLastError());

    // ---- Launch Kernel 2: Layer 2 (linear + Sigmoid) ------
    //   Grid:  N blocks  (one per sample)
    //   Block: 1 thread  (output is scalar per sample)
    dim3 grid2(N);
    dim3 block2(1);
    gpu_matmul_sigmoid_kernel<<<grid2, block2>>>(
        d_H, d_W2, d_b2, d_out, hidden_dim);
    CUDA_CHECK(cudaGetLastError());

    // ---- Synchronize and copy results back (D2H) ----------
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, sz_out, cudaMemcpyDeviceToHost));

    // ---- Free device memory --------------------------------
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_out));
}
