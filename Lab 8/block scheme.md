```mermaid
flowchart TD
    A(["Start program"]) --> B["Задать размер массива N"]
    B --> C["Выделить pinned memory (h_base, h_cpu, h_gpu, h_hyb)"]
    C --> D["Заполнить h_base случайными числами"]
    D --> E["Скопировать h_base в h_cpu, h_gpu, h_hyb"]

    %% CPU branch
    E --> F["Start CPU timer"]
    F --> G["OpenMP parallel for: i = 0..N-1"]
    G --> H["h_cpu[i] = h_cpu[i] * 2"]
    H --> I["Stop CPU timer"]

    %% GPU branch
    E --> J["cudaMalloc d_a"]
    J --> K["Start CUDA event timer (GPU)"]
    K --> L["Memcpy H->D (h_gpu -> d_a)"]
    L --> M["Launch CUDA kernel mul2"]
    M --> N["Memcpy D->H (d_a -> h_gpu)"]
    N --> O["Stop CUDA event timer (GPU)"]
    O --> P["cudaFree d_a"]

    %% Hybrid branch
    E --> Q["Разделить массив: первая половина / вторая половина"]
    Q --> R["cudaMalloc d_b (вторая половина)"]
    R --> S["Создать CUDA stream"]
    S --> T["Start CUDA event timer (Hybrid)"]

    T --> U["MemcpyAsync H->D (вторая половина)"]
    U --> V["Kernel mul2 в stream (вторая половина)"]
    V --> W["MemcpyAsync D->H (вторая половина)"]

    T --> X["CPU OpenMP: умножить первую половину"]
    X --> Y["CPU часть завершена"]

    W --> Z["StreamSynchronize"]
    Z --> AA["Stop CUDA event timer (Hybrid)"]
    AA --> AB["cudaFree d_b и stream"]

    %% Check + output
    I --> AC["Проверка: CPU vs GPU (первые 1000)"]
    O --> AC
    AA --> AC

    AC --> AD["Записать времена в results8.csv"]
    AD --> AE["Вывести результаты и время"]
    AE --> AF(["End program"])
