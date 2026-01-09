# Practical Work 4 — Optimization of GPU Parallel Code Using Memory Types (CUDA)

In this laboratory work, the task was logically divided into three parts in order to clearly demonstrate the impact of different GPU memory types on performance.  
Each part focuses on a specific approach and builds understanding step by step, from a simple baseline solution to an optimized version and performance comparison.

All implementations follow the lecture examples and use a simple, educational coding style.


## Part 1 — Sum Reduction Using Global Memory

### Description
In the first part, a naive parallel reduction of an array is implemented using only global memory.

Each GPU thread:
- reads one element from the global memory,
- adds it to a single global result using an atomic operation.

This approach is very simple and easy to understand, but it involves a large number of atomic operations and frequent access to slow global memory.

### Purpose
The goal of this part is to create a baseline solution that demonstrates how a parallel algorithm works on the GPU without any memory optimizations.

### Output
- A single integer value — the sum of all elements in the array.
- This result is used later to verify correctness of optimized versions.


## Part 2 — Sum Reduction Using Shared Memory

### Description
In the second part, the reduction algorithm is optimized by introducing shared memory.

Each block of threads:
- copies a portion of the array from global memory into shared memory,
- performs a local reduction inside the block using synchronization,
- only one thread per block adds the block sum to the global result.

This significantly reduces the number of atomic operations and global memory accesses.

### Purpose
The goal of this part is to demonstrate how shared memory can be used to optimize parallel algorithms and reduce overhead compared to the global-memory-only approach.

### Output
- A single integer value — the sum of all elements in the array.
- The result must match the output of Part 1, confirming correctness.


## Part 3 — Performance Measurement and Comparison

### Description
In the third part, the performance of both reduction approaches is measured using CUDA events.

The execution time is evaluated for three different array sizes:
- 10,000 elements
- 100,000 elements
- 1,000,000 elements

Both versions (global memory and shared memory) are executed for each size.

### Purpose
The goal of this part is to compare execution times and analyze how the choice of memory type affects performance depending on the problem size.

### Output
- A table containing:
  - array size,
  - execution time for global memory version,
  - execution time for shared memory version,
  - resulting sums for both versions.

Matching sums confirm correctness, while timing differences illustrate performance characteristics.


## Summary
By dividing the laboratory work into three logical parts, it becomes easier to:
- understand the basic principles of parallel reduction on GPU,
- observe the effect of shared memory optimization,
- analyze performance behavior for different data sizes.

This step-by-step structure provides a clear and educational comparison of GPU memory usage strategies.
