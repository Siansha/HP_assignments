// ============================================================
// cpu_nn.cpp  –  CPU (sequential) 2-layer MLP forward pass
// ============================================================

#include "cpu_nn.h"
#include <cmath>
#include <algorithm>

// ---- Activations -----------------------------------------
float relu(float x)    { return x > 0.0f ? x : 0.0f; }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---- Matrix-vector multiply ------------------------------
// y[i] = sum_j( W[i*cols + j] * x[j] ) + b[i]
void cpu_matvec(const float* W, const float* x, const float* b,
                float* y, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float acc = b[i];
        for (int j = 0; j < cols; ++j) {
            acc += W[i * cols + j] * x[j];
        }
        y[i] = acc;
    }
}

// ---- Forward pass for a single sample -------------------
float cpu_forward_one(
    const float* x,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    int input_dim, int hidden_dim, int output_dim,
    std::vector<float>& hidden_buf)
{
    // --- Layer 1: linear + ReLU ---
    cpu_matvec(W1, x, b1, hidden_buf.data(), hidden_dim, input_dim);
    for (int h = 0; h < hidden_dim; ++h)
        hidden_buf[h] = relu(hidden_buf[h]);

    // --- Layer 2: linear + Sigmoid ---
    // output_dim == 1 for binary classification
    float logit = b2[0];
    for (int h = 0; h < hidden_dim; ++h)
        logit += W2[h] * hidden_buf[h];   // W2 shape: [1 x hidden_dim]

    return sigmoid(logit);
}

// ---- Batched forward pass --------------------------------
void cpu_forward_batch(
    const float* X, int N,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    int input_dim, int hidden_dim, int output_dim,
    float* out)
{
    std::vector<float> hidden_buf(hidden_dim);
    for (int i = 0; i < N; ++i) {
        out[i] = cpu_forward_one(
            X + i * input_dim,
            W1, b1, W2, b2,
            input_dim, hidden_dim, output_dim,
            hidden_buf);
    }
}

// ---- Binary cross-entropy --------------------------------
// BCE = -( y * log(p) + (1-y) * log(1-p) ) averaged over N
float cpu_bce_loss(const float* probs, const float* labels, int N) {
    const float eps = 1e-7f;
    float total = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = std::max(eps, std::min(1.0f - eps, probs[i]));
        total += -(labels[i] * std::log(p) + (1.0f - labels[i]) * std::log(1.0f - p));
    }
    return total / N;
}

// ---- Accuracy -------------------------------------------
float cpu_accuracy(const float* probs, const float* labels, int N) {
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        int pred  = (probs[i] >= 0.5f) ? 1 : 0;
        int truth = (int)labels[i];
        if (pred == truth) ++correct;
    }
    return 100.0f * correct / N;
}
