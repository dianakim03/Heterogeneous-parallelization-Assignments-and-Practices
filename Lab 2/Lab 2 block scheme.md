```mermaid
flowchart TD
    A(start) --> B(set_sizes)
    B --> C(print_sizes)
    C --> D(loop_each_N)

    D --> E(create_base_vector)
    E --> F(fill_random)
    F --> G(print_or_skip_base)
    G --> H(make_ref_sorted)

    H --> I(bubble_block)
    I --> I1(copy_seq_par)
    I1 --> I2(time_bubble_seq)
    I2 --> I3(time_bubble_par)
    I3 --> I4(print_bubble_times)
    I4 --> I5(check_correctness)

    I5 --> J(selection_block)
    J --> J1(copy_seq_par)
    J1 --> J2(time_select_seq)
    J2 --> J3(time_select_par)
    J3 --> J4(print_select_times)
    J4 --> J5(check_correctness)

    J5 --> K(insertion_block)
    K --> K1(copy_seq_par)
    K1 --> K2(time_insert_seq)
    K2 --> K3(time_insert_par)
    K3 --> K4(print_insert_times)
    K4 --> K5(check_correctness)

    K5 --> L(next_N_or_end)
    L --> D
    L --> M(return_0)
