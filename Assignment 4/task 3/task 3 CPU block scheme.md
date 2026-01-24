```mermaid
flowchart TD
    A(["Start"]) --> B["Задать N = 1000000"]
    B --> C["Создать массив h_in"]
    C --> D["i = 0"]

    D --> E{"i < N ?"}
    E -- "Да" --> F["h_in[i] = 1"]
    F --> G["i = i + 1"]
    G --> E
    E -- "Нет" --> H["cpu_sum = 0"]

    H --> I["Запомнить cpu_start"]
    I --> J["i = 0"]

    J --> K{"i < N ?"}
    K -- "Да" --> L["cpu_sum = cpu_sum + h_in[i]"]
    L --> M["i = i + 1"]
    M --> K
    K -- "Нет" --> N["Запомнить cpu_end"]

    N --> O["Вычислить cpu_time"]
    O --> P["Перейти к GPU части"]
    P --> Q(["End CPU part"])
