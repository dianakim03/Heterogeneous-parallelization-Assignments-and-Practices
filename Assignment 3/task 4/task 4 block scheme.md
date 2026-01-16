```mermaid
flowchart TD
    A([Start]) --> B[Задает N BLOCK_BAD BLOCK_GOOD]
    B --> C[Создает массивы h_a h_b h_c на CPU]
    C --> D[Заполняет h_a и h_b значениями]
    D --> E[Выделяет память на GPU для d_a d_b d_c]
    E --> F[Копирует h_a и h_b на GPU]

    F --> G[Обнуляет d_c]
    G --> H[Запускает run_add с BLOCK_BAD]
    H --> I[Копирует результат в h_c]
    I --> J{Результат корректный}
    J -->|Да| K[ok_bad = OK]
    J -->|Нет| L[ok_bad = FAIL]

    K --> M[Обнуляет d_c]
    L --> M
    M --> N[Запускает run_add с BLOCK_GOOD]
    N --> O[Копирует результат в h_c]
    O --> P{Результат корректный}
    P -->|Да| Q[ok_good = OK]
    P -->|Нет| R[ok_good = FAIL]

    Q --> S[Выводит времена и статусы]
    R --> S

    S --> T[Освобождает память GPU]
    T --> U[Освобождает память CPU]
    U --> V([End])
