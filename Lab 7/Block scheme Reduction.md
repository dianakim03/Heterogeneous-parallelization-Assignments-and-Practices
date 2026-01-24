```mermaid
flowchart TD
    A(["Start: reduce_kernel"]) --> B["tid = threadIdx.x"]
    B --> C["gid = blockIdx.x * blockDim.x + tid"]
    C --> D["x = 0"]
    D --> E{"gid < N ?"}
    E -- "Да" --> F["x = in[gid]"]
    E -- "Нет" --> G["x = 0"]
    F --> H["sh[tid] = x"]
    G --> H
    H --> I["__syncthreads()"]

    I --> J["stride = blockDim.x / 2"]
    J --> K{"stride > 0 ?"}
    K -- "Да" --> L{"tid < stride ?"}
    L -- "Да" --> M["sh[tid] = sh[tid] + sh[tid + stride]"]
    L -- "Нет" --> N["ничего не делает"]
    M --> O["__syncthreads()"]
    N --> O
    O --> P["stride = stride / 2"]
    P --> K

    K -- "Нет" --> Q{"tid == 0 ?"}
    Q -- "Да" --> R["block_sums[blockIdx.x] = sh[0]"]
    Q -- "Нет" --> S["пропуск"]
    R --> T(["End: reduce_kernel"])
    S --> T
