```mermaid
flowchart TD
    A(["Start"]) --> B["Задать N = 1000000"]
    B --> C["Задать threadsPerBlock = 256"]
    C --> D["Вычислить blocksPerGrid"]
    D --> E["Выделить память на GPU: d_in, d_out, d_block_sums, d_block_offsets"]
    E --> F["Скопировать in на d_in"]
    F --> G["Запомнить start time GPU"]

    G --> H["Этап 1: Запуск scan_blocks_kernel (shared memory)"]

    H --> I["Внутри блока: загрузить элементы в shared memory"]
    I --> J["offset = 1"]

    J --> K{"offset < blockDim ?"}
    K -- "Да" --> L["Каждый поток берёт значение слева (если возможно)"]
    L --> M["Обновляет sh[tid] = sh[tid] + val"]
    M --> N["offset = offset * 2"]
    N --> K
    K -- "Нет" --> O["Записать out для элементов блока"]
    O --> P["Сохранить сумму блока в block_sums"]

    P --> Q["Синхронизация GPU (cudaDeviceSynchronize)"]
    Q --> R["Скопировать block_sums на CPU"]

    R --> S["Этап 2: CPU считает prefix для block_sums"]
    S --> T["block_run = 0"]
    T --> U["i = 0"]

    U --> V{"i < blocksPerGrid ?"}
    V -- "Да" --> W["block_run = block_run + block_sums[i]"]
    W --> X["block_offsets[i] = block_run"]
    X --> Y["i = i + 1"]
    Y --> V
    V -- "Нет" --> Z["Скопировать block_offsets на GPU"]

    Z --> AA["Этап 3: Запуск add_offsets_kernel"]
    AA --> AB["Для каждого элемента: взять offset предыдущих блоков"]
    AB --> AC["d_out[gid] = d_out[gid] + offset"]

    AC --> AD["Синхронизация GPU (cudaDeviceSynchronize)"]
    AD --> AE["Запомнить end time GPU"]
    AE --> AF["Скопировать d_out на CPU"]

    AF --> AG["Проверка: сравнить out_gpu и out_cpu"]
    AG --> AH{"Все элементы совпали?"}
    AH -- "Да" --> AI["Вывести: Check = OK"]
    AH -- "Нет" --> AJ["Вывести: Check = FAIL"]

    AI --> AK["Вывести последний элемент CPU и GPU"]
    AJ --> AK
    AK --> AL(["End"])
