```mermaid
flowchart TD
    A([Start]) --> B[Прочитать N, threads, reps]
    B --> C[Создать массив a[N]]
    C --> D[Заполнить массив случайными числами]

    D --> E[Прогрев вычислений]
    E --> F1[calc_seq]
    E --> F2[calc_omp]

    F1 --> G[Инициализировать t_seq = 0]
    G --> H{reps выполнены?}
    H -->|Нет| I[Замер времени calc_seq]
    I --> G
    H -->|Да| J[Среднее t_seq]

    J --> K[Инициализировать t_omp = 0]
    K --> L{reps выполнены?}
    L -->|Нет| M[parallel for + reduction]
    M --> K
    L -->|Да| N[Среднее t_omp]

    N --> O[Вычислить speedup]
    O --> P[Оценить serial_part и parallel_part]
    P --> Q[Вычислить mean, var, std]
    Q --> R[Проверить корректность суммы]
    R --> S[Вывести результаты]
    S --> T([End])
