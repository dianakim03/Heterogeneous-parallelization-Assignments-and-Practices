```mermaid
flowchart TD
    A(["Start"]) --> B["Прочитать N, iters"]
    B --> C["Создать host массив h_in[N]"]
    C --> D["cudaMalloc d_in и d_out"]
    D --> E["Memcpy H->D"]
    E --> F["Замер coalesced через cudaEvent"]
    F --> G["k_coalesced: out[i]=in[i]*2"]
    G --> H["Замер uncoalesced через cudaEvent"]
    H --> I["k_uncoalesced: i=perm(tid)"]
    I --> J["Замер shared_fix через cudaEvent"]
    J --> K["k_shared_fix: coalesced load -> shared -> perm read"]
    K --> L["Проверка: сравнить первые 1000 элементов"]
    L --> M["Вывести времена и check"]
    M --> N["cudaFree d_in/d_out"]
    N --> O(["End"])
