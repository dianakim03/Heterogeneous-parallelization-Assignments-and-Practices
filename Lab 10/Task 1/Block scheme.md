```mermaid
flowchart TD
A([Start]);
B[Read N, threads, reps];
C[Create array a_N];
D[Fill array with random values];
E[Warm up seq and omp];
F[Set t_seq = 0];
G{Reps done for seq?};
H[Run calc_seq and add time];
I[Compute avg t_seq];
J[Set t_omp = 0];
K{Reps done for omp?};
L[Run calc_omp with reduction];
M[Compute avg t_omp];
N[Compute speedup];
O[Estimate serial and parallel parts];
P[Compute mean var stddev];
Q[Check correctness];
R[Print results];
S([End]);

A --> B;
B --> C;
C --> D;
D --> E;
E --> F;
F --> G;
G -- No --> H;
H --> G;
G -- Yes --> I;
I --> J;
J --> K;
K -- No --> L;
L --> K;
K -- Yes --> M;
M --> N;
N --> O;
O --> P;
P --> Q;
Q --> R;
R --> S;

