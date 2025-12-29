flowchart TD
    A(start) --> B(set_sizes)
    B --> C(print_sizes)
    C --> D(loop_each_N)

    D --> E(create_base_vector)
    E --> F(fill_random)
    F --> G(copy_for_cpu_gpu)

    G --> H(cpu_merge_block)
    H --> H1(copy_cpu)
    H1 --> H2(time_cpu_merge)
    H2 --> H3(check_cpu_merge)

    H3 --> I(cpu_quick_block)
    I --> I1(copy_cpu)
    I1 --> I2(time_cpu_quick)
    I2 --> I3(check_cpu_quick)

    I3 --> J(cpu_heap_block)
    J --> J1(copy_cpu)
    J1 --> J2(time_cpu_heap)
    J2 --> J3(check_cpu_heap)

    J3 --> K(gpu_merge_block)
    K --> K1(copy_to_gpu)
    K1 --> K2(time_gpu_merge)
    K2 --> K3(copy_from_gpu)
    K3 --> K4(check_gpu_merge)

    K4 --> L(gpu_quick_block)
    L --> L1(copy_to_gpu)
    L1 --> L2(time_gpu_quick)
    L2 --> L3(copy_from_gpu)
    L3 --> L4(check_gpu_quick)

    L4 --> M(gpu_heap_block)
    M --> M1(copy_to_gpu)
    M1 --> M2(time_gpu_heap)
    M2 --> M3(copy_from_gpu)
    M3 --> M4(check_gpu_heap)

    M4 --> N(print_results)
    N --> O(next_N_or_end)
    O --> D
    O --> P(return_0)
