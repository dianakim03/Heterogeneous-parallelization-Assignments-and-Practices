```mermaid
flowchart TD
    A(["Start"]) --> B["Задать N"]
    B --> C["Создать массив arr"]
    C --> D["i = 0"]

    D --> E{"i < N ?"}
    E -- "Да" --> F["arr[i] = 1"]
    F --> G["i = i + 1"]
    G --> E
    E -- "Нет" --> H["sum = 0"]

    H --> I["Запомнить start time"]
    I --> J["i = 0"]

    J --> K{"i < N ?"}
    K -- "Да" --> L["sum = sum + arr[i]"]
    L --> M["i = i + 1"]
    M --> K
    K -- "Нет" --> N["Запомнить end time"]

    N --> O["Вычислить duration"]
    O --> P["Вывести sum и time"]
    P --> Q(["End"])
