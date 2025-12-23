```mermaid
flowchart TD
    A(start) --> B(set_N_1000000)
    B --> C(allocate_array)
    C --> D(init_random_seed)
    D --> E(fill_array_random)

    E --> F(start_seq_timer)
    F --> G(init_seq_min_max)
    G --> H(find_seq_min_max_loop)
    H --> I(stop_seq_timer)
    I --> J(save_seq_time)

    J --> K(start_par_timer)
    K --> L(init_par_min_max)

    L --> M(omp_parallel_region)
    M --> N(init_local_min_max)
    N --> O(omp_for_scan)
    O --> P(omp_critical_update)
    P --> Q(stop_par_timer)
    Q --> R(save_par_time)

    R --> S(print_all_results)
    S --> T(print_threads)
    T --> U(free_memory)
    U --> V(return_0)
