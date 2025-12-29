```mermaid
flowchart TD
    A(start) --> B(cuda_warmup)
    B --> C(run_N_10000)
    C --> D(run_N_100000)
    D --> E(return_0)

    %% runOne(n)
    C --> C1(create_host_vector_h)
    C1 --> C2(fill_random_values)
    C2 --> C3(cudaMalloc_d_in_d_tmp)
    C3 --> C4(cudaMemcpy_H2D)
    C4 --> C5(call_gpuMergeSort)
    C5 --> C6(cudaMemcpy_D2H)
    C6 --> C7(check_is_sorted)
    C7 --> C8(print_time_and_status)
    C8 --> C9(cudaFree_d_in_d_tmp)

    %% gpuMergeSort(d_in, d_tmp, n)
    C5 --> G1(create_cuda_events)
    G1 --> G2(record_start)
    G2 --> G3(calc_blocks_for_chunks)
    G3 --> G4(kernel_sortChunksBitonic)
    G4 --> G5(sync_after_chunk_sort)
    G5 --> G6(init_src_dst_pointers)
    G6 --> G7(loop_width_doubles)

    %% merge loop
    G7 --> H1(calc_pairs_for_merge)
    H1 --> H2(kernel_mergePass)
    H2 --> H3(sync_after_merge)
    H3 --> H4(swap_src_dst)
    H4 --> G7

    %% finish
    G7 --> G8(if_src_not_d_in_copy_back)
    G8 --> G9(record_end)
    G9 --> G10(elapsed_time_ms)
    G10 --> G11(destroy_events)
    G11 --> C6
