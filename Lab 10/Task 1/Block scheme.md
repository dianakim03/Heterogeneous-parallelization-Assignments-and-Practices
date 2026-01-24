```mermaid
flowchart TD
    A(["Start"]) --> B["Прочитать N, threads, reps"]
    B --> C["Создать массив a[N]"]
    C --> D["Заполнить случайными числами"]
    D --> E["Прогрев: calc_seq и calc_omp"]
    E --> F["Замер seq: reps раз"]
    F --> G["calc_seq: sum и sumsq"]
    G --> H["Замер omp: reps раз"]
    H --> I["calc_omp: parallel for + reduction(sum,sumsq)"]
    I --> J["speedup = t_seq / t_omp"]
    J --> K["Оценить serial_part f и parallel_part"]
    K --> L["Посчитать mean/var/std из sum и sumsq"]
    L --> M["Вывести результаты"]
    M --> N(["End"])
