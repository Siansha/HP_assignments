#pragma once
// ============================================================
// cpu_nn.h  –  CPU (sequential) 2-layer MLP forward pass
// ============================================================
// Network architecture:
//   Input  (input_dim)
//     → Dense Layer 1 (hidden_dim)  + ReLU
//     → Dense Layer 2 (output_dim)  + Sigmoid
//   Loss: Binary Cross-Entropy
//
// Everything is row-major.  Weights are stored transposed
// to allow a clean dot-product loop:
//   W1: [hidden_dim x input_dim]
//   b1: [hidden_dim]
//   W2: [output_dim x hidden_dim]
//   b2: [output_dim]

#include <vector>

// ---- Activation functions --------------------------------
float relu(float x);
float sigmoid(float x);

// ---- Matrix-vector multiply (CPU) -----------------------
// y = W * x + b
//   W: [rows x cols]  x: [cols]  b: [rows]  y: [rows]
void cpu_matvec(const float* W, const float* x, const float* b,
                float* y, int rows, int cols);

// ---- Full forward pass for one sample -------------------
// Returns the sigmoid output probability for class 1.
float cpu_forward_one(
    const float* x,            // input vector [input_dim]
    const float* W1, const float* b1,   // layer 1 weights/bias
    const float* W2, const float* b2,   // layer 2 weights/bias
    int input_dim, int hidden_dim, int output_dim,
    std::vector<float>& hidden_buf);    // scratch buffer [hidden_dim]

// ---- Batched forward pass --------------------------------
// Runs cpu_forward_one for every sample in X.
// X:    [N x input_dim]  row-major
// out:  [N]  predicted probabilities
void cpu_forward_batch(
    const float* X, int N,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    int input_dim, int hidden_dim, int output_dim,
    float* out);

// ---- Binary cross-entropy loss --------------------------
float cpu_bce_loss(const float* probs, const float* labels, int N);

// ---- Accuracy (threshold 0.5) ---------------------------
float cpu_accuracy(const float* probs, const float* labels, int N);
