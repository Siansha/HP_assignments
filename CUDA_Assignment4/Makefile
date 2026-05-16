# ============================================================
# Makefile  –  CUDA Neural Network Benchmark
# ============================================================
# Usage:
#   make          – compile everything
#   make run      – compile and run
#   make clean    – remove build artefacts

NVCC      = nvcc
CXX       = g++
NVCCFLAGS = -O2 -std=c++14 -arch=120    # change sm_75 to match your GPU
CXXFLAGS  = -O2 -std=c++14

# ---- Object files ----------------------------------------
OBJS = main.o cpu_nn.o utils.o gpu_nn.o

TARGET = nn_benchmark

# ---- Default target --------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# ---- Compilation rules -----------------------------------
main.o: main.cpp utils.h cpu_nn.h gpu_nn.h
	$(NVCC) $(NVCCFLAGS) -c main.cpp -o main.o

cpu_nn.o: cpu_nn.cpp cpu_nn.h
	$(NVCC) $(NVCCFLAGS) -c cpu_nn.cpp -o cpu_nn.o

utils.o: utils.cpp utils.h
	$(NVCC) $(NVCCFLAGS) -c utils.cpp -o utils.o

gpu_nn.o: gpu_nn.cu gpu_nn.h
	$(NVCC) $(NVCCFLAGS) -c gpu_nn.cu -o gpu_nn.o

# ---- Convenience targets ---------------------------------
run: all
	./$(TARGET)

clean:
	rm -f *.o $(TARGET)

.PHONY: all run clean
