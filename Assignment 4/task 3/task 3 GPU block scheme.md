```mermaid
flowchart TD
    A(["Start GPU part"]) --> B["Выделить память на GPU: d_in и d_block_sums"]
    B --> C["Скопировать h_in на d_in"]
    C --> D["Задать threadsPerBlock = 256"]
    D --> E["Вычислить blocksPerGrid"]
    E --> F["Создать CUDA events e_start и e_end"]
    F --> G["Записать e_start"]

    G --> H["Запустить reduction_kernel (shared memory)"]

    H --> I["В kernel: вычислить gid и tid"]
    I --> J{"gid < N ?"}
    J -- "Да" --> K["x = in[gid]"]
    J -- "Нет" --> L["x = 0"]
    K --> M["sh[tid] = x"]
    L --> M
    M --> N["__syncthreads"]

    N --> O["stride = blockDim / 2"]
    O --> P{"stride > 0 ?"}
    P -- "Да" --> Q{"tid < stride ?"}
    Q -- "Да" --> R["sh[tid] = sh[tid] + sh[tid + stride]"]
    Q -- "Нет" --> S["Пропуск"]
    R --> T["__syncthreads"]
    S --> T
    T --> U["stride = stride / 2"]
    U --> P
    P -- "Нет" --> V{"tid == 0 ?"}
    V -- "Да" --> W["block_sums[blockIdx] = sh[0]"]
    V -- "Нет" --> X["Пропуск"]

    W --> Y["cudaDeviceSynchronize"]
    X --> Y

    Y --> Z["Записать e_end и синхронизировать"]
    Z --> AA["Вычислить gpu_ms по events"]
    AA --> AB["Скопировать d_block_sums на CPU"]

    AB --> AC["gpu_sum = 0"]
    AC --> AD["i = 0"]

    AD --> AE{"i < blocksPerGrid ?"}
    AE -- "Да" --> AF["gpu_sum = gpu_sum + h_block_sums[i]"]
    AF --> AG["i = i + 1"]
    AG --> AE
    AE -- "Нет" --> AH["Вывести cpu_sum, gpu_sum, cpu_time, gpu_ms"]

    AH --> AI["Освободить память GPU и удалить events"]
    AI --> AJ(["End GPU part"])
