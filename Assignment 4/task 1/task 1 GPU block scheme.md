```mermaid
flowchart TD
    A(["Start"]) --> B["Задать N"]
    B --> C["Выделить unified memory для массива"]
    C --> D["i = 0"]

    D --> E{"i < N ?"}
    E -- "Да" --> F["arr[i] = 1"]
    F --> G["i = i + 1"]
    G --> E
    E -- "Нет" --> H["Задать threadsPerBlock"]

    H --> I["Вычислить blocksPerGrid"]
    I --> J["Запомнить start time"]
    J --> K["Запуск CUDA kernel"]

    K --> L["Вычислить idx"]
    L --> M{"idx < N ?"}
    M -- "Да" --> N["arr[idx] = 2"]
    M -- "Нет" --> O["Пропуск"]
    N --> P["cudaDeviceSynchronize"]
    O --> P

    P --> Q["Запомнить end time"]
    Q --> R["sum = 0"]
    R --> S["i = 0"]

    S --> T{"i < N ?"}
    T -- "Да" --> U["sum = sum + arr[i]"]
    U --> V["i = i + 1"]
    V --> T
    T -- "Нет" --> W["Вычислить duration"]

    W --> X["Вывести sum и time"]
    X --> Y["Освободить память"]
    Y --> Z(["End"])
