```mermaid
flowchart TD
    A(["Start"]) --> B["MPI_Init"]
    B --> C["Получить rank (номер процесса)"]
    C --> D["Получить size (кол-во процессов)"]
    D --> E["Задать N = 1000000"]
    E --> F["local_n = N / size"]
    F --> G["Создать local_data[local_n]"]

    G --> H{"rank == 0 ?"}
    H -- "Да" --> I["Создать data[N]"]
    I --> J["i = 0"]
    J --> K{"i < N ?"}
    K -- "Да" --> L["data[i] = 1"]
    L --> M["i = i + 1"]
    M --> K
    K -- "Нет" --> N["Готово: data заполнен"]
    H -- "Нет" --> O["data не создаётся"]

    N --> P["MPI_Barrier (синхронизация)"]
    O --> P

    P --> Q["Запомнить start time"]
    Q --> R["MPI_Scatter (раздать куски массива)"]

    R --> S["local_sum = 0"]
    S --> T["i = 0"]
    T --> U{"i < local_n ?"}
    U -- "Да" --> V["local_sum = local_sum + local_data[i]"]
    V --> W["i = i + 1"]
    W --> U
    U -- "Нет" --> X["MPI_Reduce (собрать суммы в global_sum на rank 0)"]

    X --> Y["MPI_Barrier (синхронизация)"]
    Y --> Z["Запомнить end time"]
    Z --> AA["duration = end - start"]

    AA --> AB{"rank == 0 ?"}
    AB -- "Да" --> AC["Вывести size, global_sum, duration"]
    AB -- "Нет" --> AD["Не выводит"]

    AC --> AE["MPI_Finalize"]
    AD --> AE
    AE --> AF(["End"])
