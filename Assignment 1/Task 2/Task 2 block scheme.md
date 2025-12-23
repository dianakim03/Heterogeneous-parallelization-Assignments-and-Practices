```mermaid
flowchart TD
    A(start) --> B(init_random_seed)
    B --> C(set_N_1000000)
    C --> D(allocate_array)
    D --> E(fill_array_random)

    E --> F(start_timer)
    F --> G(init_min_max)
    G --> H(find_min_max_loop)
    H --> I(stop_timer)

    I --> J(print_min_max)
    J --> K(print_time)
    K --> L(free_memory)
    L --> M(return_0)
