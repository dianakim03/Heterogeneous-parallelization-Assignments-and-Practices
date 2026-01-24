```mermaid
flowchart TD
    A(["Start CPU"]) --> B["Создать массив N"]
    B --> C["Заполнить массив"]
    C --> D["Start timer"]
    D --> E["OpenMP parallel for: i=0..N-1"]
    E --> F["a[i] = a[i] * 2"]
    F --> G["End timer"]
    G --> H["Вывести время CPU"]
    H --> I(["End CPU"])
