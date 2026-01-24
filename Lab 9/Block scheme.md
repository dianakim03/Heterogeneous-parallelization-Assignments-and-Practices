```mermaid
flowchart TD
    A(["Start"]) --> B["MPI_Init"]
    B --> C["MPI_Comm_rank / MPI_Comm_size"]
    C --> D["Считать аргументы: task и N"]
    D --> E{task == 1?}

    %% TASK 1
    E -->|Да| T1A["rank0: создать массив N случайных double"]
    T1A --> T1B["Посчитать counts/displs для Scatterv (с остатком)"]
    T1B --> T1C["MPI_Scatterv: раздать части массива"]
    T1C --> T1D["Каждый процесс: local_sum и local_sumsq"]
    T1D --> T1E["MPI_Reduce: собрать global_sum и global_sumsq на rank0"]
    T1E --> T1F["rank0: mean и stddev + вывести время"]
    T1F --> Z["MPI_Finalize"]
    
    %% TASK 2 decision
    E -->|Нет| F{task == 2?}

    %% TASK 2
    F -->|Да| T2A["rank0: создать A(NxN) и b(N)"]
    T2A --> T2B["MPI_Scatter: раздать строки A и куски b"]
    T2B --> T2C["Цикл k=0..N-1: owner формирует pivot"]
    T2C --> T2D["MPI_Bcast: разослать pivot строку всем"]
    T2D --> T2E["Каждый процесс: прямой ход для своих строк"]
    T2E --> T2F["MPI_Gather: собрать A и b на rank0"]
    T2F --> T2G["rank0: обратный ход -> x"]
    T2G --> T2H["rank0: вывести первые элементы x и время"]
    T2H --> Z["MPI_Finalize"]

    %% TASK 3 decision
    F -->|Нет| G{task == 3?}

    %% TASK 3
    G -->|Да| T3A["rank0: создать матрицу графа G(NxN)"]
    T3A --> T3B["MPI_Scatter: раздать строки G"]
    T3B --> T3C["MPI_Allgather: собрать all_mat у всех"]
    T3C --> T3D["Цикл k=0..N-1: обновить локальные строки"]
    T3D --> T3E["MPI_Allgather: обмен обновлёнными строками"]
    T3E --> T3F["MPI_Gather: собрать итог на rank0"]
    T3F --> T3G["rank0: вывести 8x8 и время"]
    T3G --> Z["MPI_Finalize"]

    %% wrong task
    G -->|Нет| W["rank0: вывести ошибку (task должен быть 1/2/3)"]
    W --> Z["MPI_Finalize"]
    Z --> END(["End"])
