```mermaid
flowchart TD
    A([Start]) --> B[Задает N BLOCK STRIDE]
    B --> C[Создает массив h_a на CPU]
    C --> D[Заполняет h_a значениями i]

    D --> E[Выделяет память на GPU для d_a]
    E --> F[Копирует h_a на GPU в d_a]

    F --> G[Запускает run_coalesced]
    G --> H[Старт cudaEvent]
    H --> I[Запускает kernel_coalesced]
    I --> J[Стоп cudaEvent и получает t_coal]

    J --> K[Копирует d_a обратно в h_a]
    K --> L{Результат корректный}
    L -->|Да| M[ok_coal = OK]
    L -->|Нет| N[ok_coal = FAIL]

    M --> O[Восстанавливает h_a = i]
    N --> O

    O --> P[Копирует h_a снова на GPU в d_a]

    P --> Q[Запускает run_noncoalesced]
    Q --> R[Старт cudaEvent]
    R --> S[Запускает kernel_noncoalesced]
    S --> T[Стоп cudaEvent и получает t_non]

    T --> U[Копирует d_a обратно в h_a]
    U --> V{Результат корректный}
    V -->|Да| W[ok_non = OK]
    V -->|Нет| X[ok_non = FAIL]

    W --> Y[Выводит N BLOCK STRIDE t_coal t_non]
    X --> Y

    Y --> Z[Освобождает память GPU и CPU]
    Z --> AA([End])
