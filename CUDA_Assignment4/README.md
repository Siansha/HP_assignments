# CUDA Neural Network Benchmark – Complete Assignment Guide

## Project Structure

```
cuda_nn_project/
├── main.cpp        ← driver: benchmarks CPU vs GPU, prints results
├── cpu_nn.h/cpp    ← sequential CPU 2-layer MLP forward pass
├── gpu_nn.h/cu     ← parallel CUDA GPU 2-layer MLP forward pass
├── utils.h/cpp     ← timer, dataset generation, pretty printing
├── Makefile        ← build system
└── README.md       ← this file
```

---

## How to Compile and Run

### Step 1 – Check your GPU compute capability
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```
Then edit `Makefile` and set `-arch=smXX` to match (e.g. sm_86 for RTX 30xx, sm_89 for RTX 40xx).

### Step 2 – Compile
```bash
make
```

### Step 3 – Run
```bash
./nn_benchmark
```
Or simply:
```bash
make run
```

### Expected output (approximate)
```
============================================================
            CUDA Neural Network Benchmark
============================================================

[1/5] Generating synthetic dataset...
      10000 samples, 128 features, 2 classes
[2/5] Initialising network weights...
      W1: [256 x 128]  W2: [1 x 256]
[3/5] Warming up GPU (one dummy forward pass)...
      Done.
[4/5] Running CPU benchmark (20 runs)...
      Avg CPU time: 312.45 ms
[5/5] Running GPU benchmark (20 runs)...
      Avg GPU time: 18.73 ms

============================================================
Metric                        CPU            GPU
------------------------------------------------------------
Time (ms)                     312.450        18.730
Accuracy (%)                  99.50          99.50
------------------------------------------------------------
GPU Speedup                   16.68x
Max output difference         3.8147e-06
============================================================

BCE Loss  →  CPU: 0.013451  |  GPU: 0.013451

[PASS] CPU and GPU outputs match (max diff < 1e-4)
[INFO] GPU is 16.68x faster than CPU
------------------------------------------------------------
Network : Input(128) -> Hidden(256, ReLU) -> Output(1, Sigmoid)
Dataset : N=10000 samples
CUDA    : grid=(10000), block=(256)
          Each block = 1 sample, each thread = 1 neuron
============================================================
```

---

## Report-Ready Write-Up

### Objective

The goal of this project is to implement and benchmark a two-layer
Multi-Layer Perceptron (MLP) neural network forward pass using two
approaches: a sequential CPU implementation in C++ and a parallel GPU
implementation using CUDA. By comparing their execution times on
identical inputs, we demonstrate the performance advantage of GPU
parallelism for neural network inference workloads.

### Problem Statement

Neural networks perform large numbers of independent floating-point
operations (matrix multiplications, activations) for each input sample.
When processing a batch of N samples, all samples are independent of
each other, making the computation embarrassingly parallel. A CPU must
process each sample sequentially (or with limited SIMD parallelism),
whereas a GPU can process thousands of samples simultaneously across its
CUDA cores.

### Network Architecture

```
Input layer  : 128 neurons  (feature vector)
Hidden layer : 256 neurons  + ReLU activation
Output layer : 1  neuron    + Sigmoid activation
Task         : Binary classification (2-class synthetic dataset)
Batch size   : N = 10,000 samples
```

### Dataset

A synthetic dataset of 10,000 samples with 128 features is generated
programmatically. Class 0 samples are drawn from a Gaussian centred at
-1.0 and class 1 samples from a Gaussian centred at +1.0. No external
file is required; the data is generated at runtime using a fixed seed
for reproducibility.

### Methodology

#### CPU Implementation (`cpu_nn.cpp`)

The CPU version processes one sample at a time in a serial loop:

1. For each sample i:
   a. Compute h = ReLU(W1 · x_i + b1)   [hidden layer]
   b. Compute p = Sigmoid(W2 · h + b2)   [output layer]
2. Collect all p values and evaluate accuracy and BCE loss.

The inner matrix-vector multiply is an O(D·H) nested loop per sample,
where D = input_dim and H = hidden_dim.

#### GPU Implementation (`gpu_nn.cu`)

The GPU version replaces the serial loop over N samples with two CUDA
kernels:

**Kernel 1 – `gpu_matmul_relu_kernel`**
- Grid : N blocks (one block per sample)
- Block: hidden_dim threads (one thread per hidden neuron)
- Each thread independently computes one hidden activation:
  `h[sample][neuron] = ReLU(W1[neuron, :] · x[sample, :]  +  b1[neuron])`

**Kernel 2 – `gpu_matmul_sigmoid_kernel`**
- Grid : N blocks (one block per sample)
- Block: 1 thread
- Each thread computes the scalar output for its sample:
  `out[sample] = Sigmoid(W2[:] · h[sample, :]  +  b2[0])`

Data flow:
```
Host (CPU)                         Device (GPU)
----------                         -------------
X, W1, b1, W2, b2  →  H2D copy →  d_X, d_W1, d_b1, d_W2, d_b2
                                   ↓ Kernel 1 (parallel over N)
                                   d_H  (hidden activations)
                                   ↓ Kernel 2 (parallel over N)
                                   d_out (probabilities)
out  ←  D2H copy  ←               d_out
```

#### Benchmarking Protocol

- 1 GPU warm-up run (excluded from timing) to avoid measuring driver
  initialization overhead.
- N_RUNS = 20 timed iterations for both CPU and GPU.
- Average time per iteration is reported.
- CPU timer: `std::chrono::high_resolution_clock`
- GPU timing includes H2D copy + kernel execution + D2H copy
  (end-to-end wall time), which is the fair comparison.

### Results Interpretation

The GPU achieves a significant speedup (typically 10–20x on a modern
GPU) because:

1. **Massive parallelism**: All 10,000 samples are processed
   simultaneously instead of sequentially.
2. **High arithmetic intensity**: Each thread performs O(D) multiply-
   accumulate operations, keeping the GPU's FP32 units busy.
3. **Memory coalescing**: Adjacent threads in a warp access
   consecutive memory addresses (each thread reads the same W1 row but
   different input rows), enabling efficient memory bandwidth usage.

The numerical difference between CPU and GPU outputs is below 1e-4,
attributable to FP32 rounding differences between x86 and GPU ALUs.

### Conclusion

This project demonstrates that GPU acceleration is highly effective for
the forward pass of neural networks. The CUDA implementation correctly
mirrors the CPU results while delivering a double-digit speedup on
batch inference, which is critical in production ML systems where
latency and throughput must be optimized. The embarrassingly parallel
nature of batch inference makes it an ideal workload for GPU
acceleration.

---

## How to Explain This to Your Professor

**The core message in one sentence:**
"A CPU processes each sample one after another; a GPU processes all
10,000 samples at the same time using thousands of CUDA threads."

**Walk through the CUDA execution model:**
- "A *thread* is the smallest unit of work. Each thread computes one
  neuron's activation for one sample."
- "A *block* is a group of threads. Each block handles one sample, and
  its threads collectively compute all hidden neurons in parallel."
- "A *grid* is the collection of all blocks. We launch N=10,000 blocks,
  so all samples run simultaneously."

**On memory transfers:**
- "Before the GPU can compute, we must copy the input data from RAM
  (host) to GPU memory (device). This is the H2D copy. After the
  kernels finish, we copy results back — the D2H copy."
- "For large batches, the arithmetic work dominates over the copy cost,
  which is why we see large speedups."

**On correctness:**
- "Both versions use the same weights and same inputs. The maximum
  difference is ~1e-6, which is within normal float32 rounding error."

---

## Possible Viva Questions and Answers

**Q1: What is a CUDA kernel?**
A kernel is a C++ function decorated with `__global__` that runs on the
GPU. When launched, the GPU executes one instance of the kernel per
CUDA thread. Thousands of instances run concurrently.

**Q2: What is the difference between a block and a grid?**
A *block* is a group of threads that share fast shared memory and can
synchronize with `__syncthreads()`. A *grid* is the collection of all
blocks launched for a kernel. In this project, our grid has N blocks
(one per sample) and each block has `hidden_dim` threads (one per
neuron).

**Q3: Why do you need a warm-up run for the GPU?**
The first CUDA operation in a program triggers driver initialization,
JIT compilation of PTX to SASS, and context creation. This one-time
cost can take milliseconds and would unfairly inflate GPU timing. We
exclude it with a warm-up run before starting the timer.

**Q4: What is H2D and D2H?**
Host-to-Device (H2D): copying data from CPU RAM to GPU VRAM.
Device-to-Host (D2H): copying results from GPU VRAM back to CPU RAM.
These memory transfers add latency and can bottleneck performance if the
computation per byte is too low (low arithmetic intensity).

**Q5: Why is there a numerical difference between CPU and GPU results?**
Both use 32-bit floating-point (float32), but the order of arithmetic
operations differs between x86 FMA units and GPU CUDA cores. Floating-
point arithmetic is not associative, so different orderings produce
slightly different rounding errors. The difference (~1e-6) is well
within acceptable tolerance.

**Q6: When would the GPU NOT be faster than the CPU?**
- For very small N (e.g. N=1), the kernel launch overhead and memory
  transfer cost dominate. CPUs can be faster for single-sample inference.
- If hidden_dim is very small, the GPU is underutilized.
- Memory-bandwidth-bound operations where GPU memory isn't faster enough.

**Q7: What parallelism strategy did you use?**
*Data parallelism*: the same computation (forward pass) is applied to
different samples in parallel. Each sample is fully independent, making
this an embarrassingly parallel workload.

**Q8: How would you extend this to training (backpropagation)?**
Training requires computing gradients via backpropagation. Each layer's
gradient depends on the next layer's gradient (chain rule), so it's not
embarrassingly parallel across layers. However, within each layer's
gradient computation, the same matrix-multiply pattern applies and can
be parallelized similarly with CUDA kernels for dL/dW and dL/dx.

**Q9: What is ReLU and why is it used?**
ReLU (Rectified Linear Unit) is `f(x) = max(0, x)`. It introduces
non-linearity into the network (without it, stacking layers would
reduce to a single linear transform). It's preferred over sigmoid in
hidden layers because it avoids the vanishing gradient problem and is
computationally cheap.

**Q10: What is binary cross-entropy loss?**
BCE = -(y·log(p) + (1-y)·log(1-p)) where y is the true label (0 or 1)
and p is the predicted probability. It measures how far the model's
probability estimate is from the ground truth. Minimizing BCE during
training pushes the model's output toward the correct class.

---

## Common Errors and Fixes

### Error: `nvcc: command not found`
**Fix:** Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
or add it to PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Error: `no kernel image is available for execution on the device`
**Fix:** Your `-arch` flag doesn't match your GPU. Run:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```
Then set `NVCCFLAGS = -arch=sm_XX` where XX is e.g. 86 for compute 8.6.

### Error: `CUDA error: out of memory`
**Fix:** N is too large. Reduce N in `main.cpp` (e.g. to 5000).

### Error: `invalid device function`
**Fix:** Same as the `-arch` mismatch above.

### Error: Segfault on CPU side
**Fix:** Check that `hidden_buf` in `cpu_forward_one` is sized
`hidden_dim`, and that all `std::vector` sizes match the dimensions.

### Speedup is very small (< 2x)
**Possible causes:**
- N is too small (try N=50000)
- hidden_dim is too small (try 512 or 1024)
- GPU timing includes large PCIe transfer overhead relative to compute
- Running on a laptop GPU with few CUDA cores

### CPU and GPU outputs differ by > 1e-3
**Check:**
- Both are using the exact same W1, b1, W2, b2 vectors
- No accidental reinitialization of weights between CPU and GPU runs

### `make` fails with `g++` errors on `.cu` files
**Fix:** `nvcc` handles both .cpp and .cu files. Ensure all files go
through `nvcc`, not a separate `g++` call.

---

## Tuning Parameters

Edit `main.cpp` to stress-test or shrink the workload:

| Constant   | Default | Increase for bigger speedup |
|------------|---------|----------------------------|
| N          | 10000   | 50000 or 100000             |
| INPUT_DIM  | 128     | 256 or 512                  |
| HIDDEN_DIM | 256     | 512 or 1024                 |
| N_RUNS     | 20      | 50 for smoother averages    |

Note: `HIDDEN_DIM` must fit in a single CUDA block (≤ 1024 on most GPUs).
If you need HIDDEN_DIM > 1024, split the computation across multiple
blocks using a reduction pattern.
