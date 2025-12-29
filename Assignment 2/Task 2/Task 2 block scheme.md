```mermaid
flowchart TD
    A(start) --> B(set_N_10000)
    B --> C(allocate_array)
    C --> D(init_random)
    D --> E(fill_array)

    %% SEQUENTIAL PART
    E --> F(seq_start_time)
    F --> G(seq_init_min_max)
    G --> H(seq_loop_find_min_max)
    H --> I(seq_end_time)

    %% PARALLEL PART
    I --> J(par_start_time)
    J --> K(init_global_min_max)
    K --> L(omp_parallel_region)
    L --> M(local_min_max)
    M --> N(omp_for_loop)
    N --> O(omp_critical_update)
    O --> P(par_end_time)

    %% OUTPUT
    P --> Q(print_results)
    Q --> R(free_memory)
    R --> S(return_0)
