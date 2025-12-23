```mermaid
flowchart TD
    A(start) --> B(input_N)
    B --> C{N_gt_0}
    C -- no --> D(return_1)
    C -- yes --> E(allocate_array)

    E --> F(fill_array_random)
    F --> G(check_print_limit)
    G --> H(print_array_or_skip)

    H --> I(start_seq_timer)
    I --> J(calc_seq_average)
    J --> K(stop_seq_timer)

    K --> L(start_par_timer)
    L --> M(calc_par_average_reduction)
    M --> N(stop_par_timer)

    N --> O(print_results)
    O --> P(check_results_equal)
    P --> Q(print_threads)

    Q --> R(free_memory)
    R --> S(return_0)
