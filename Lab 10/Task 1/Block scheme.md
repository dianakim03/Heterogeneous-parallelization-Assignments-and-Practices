```mermaid
flowchart TD
    A([Start])
    B[Read N, threads, reps]
    C[Create array a[N]]
    D[Fill array with random values]
    E[Warm-up: calc_seq and calc_omp]
    F[Initialize t_seq]
    G{Reps finished?}
    H[Run calc_seq and measure time]
    I[Compute average t_seq]
    J[Initialize t_omp]
    K{Reps finished?}
    L[Run calc_omp with reduction]
    M[Compute average t_omp]
    N[Compute speedup]
    O[Estimate serial and parallel parts]
    P[Compute mean, variance, stddev]
    Q[Check correctness]
    R[Print results]
    S([End])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -- No --> H
    H --> G
    G -- Yes --> I
    I --> J
    J --> K
    K -- No --> L
    L --> K
    K -- Yes --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
