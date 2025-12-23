```mermaid
flowchart TD
    A(start) --> B(init_random_seed)
    B --> C(set_N_50000)
    C --> D(allocate_array)
    D --> E(init_sum)
    E --> F(fill_array_and_sum)
    F --> G(calc_average)
    G --> H(print_average)
    H --> I(free_memory)
    I --> J(return_0)
