```mermaid
flowchart TD
    A(["Start"]) --> B["Задать N = 1000000"]
    B --> C["Создать массив in и out_cpu"]
    C --> D["Заполнить in значением 1"]
    D --> E["run = 0"]
    E --> F["i = 0"]

    F --> G{"i < N ?"}
    G -- "Да" --> H["run = run + in[i]"]
    H --> I["out_cpu[i] = run"]
    I --> J["i = i + 1"]
    J --> G

    G -- "Нет" --> K["Вывести время CPU"]
    K --> L(["End"])
