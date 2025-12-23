```mermaid
flowchart TD
    A(start) --> B(set_N_5000000)
    B --> C(allocate_array)
    C --> D(init_random_seed)
    D --> E(fill_array_random)

    E --> F(start_seq_timer)
    F --> G(sum_seq_loop)
    G --> H(calc_avg_seq)
    H --> I(stop_seq_timer)
    I --> J(save_seq_time)

    J --> K(start_par_timer)
    K --> L(sum_par_reduction)
    L --> M(calc_avg_par)
    M --> N(stop_par_timer)
    N --> O(save_par_time)

    O --> P(print_results)
    P --> Q(print_threads)
    Q --> R(free_memory)
    R --> S(return_0)
