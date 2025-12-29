```mermaid
flowchart TD
    A(start) --> B(set_sizes_1000_10000)
    B --> C(loop_each_N)

    C --> D(allocate_arr1_arr2)
    D --> E(init_random_seed)
    E --> F(fill_arrays_same_values)

    %% SEQUENTIAL SELECTION SORT
    F --> G(seq_start_time)
    G --> H(call_selectionSortSequential)
    H --> I(seq_end_time)

    %% PARALLEL SELECTION SORT
    I --> J(par_start_time)
    J --> K(call_selectionSortParallel)
    K --> K1(loop_i_outer)
    K1 --> K2(omp_parallel_for_search_min)
    K2 --> K3(critical_update_minIndex)
    K3 --> K4(swap_elements)
    K4 --> L(par_end_time)

    %% OUTPUT + CLEANUP
    L --> M(print_times_threads)
    M --> N(free_memory)
    N --> O(next_N_or_end)
    O --> C
    O --> P(return_0)
