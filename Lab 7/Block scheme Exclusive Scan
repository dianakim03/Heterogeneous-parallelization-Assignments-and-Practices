```mermaid
flowchart TD
    A(["Start: scan_block_exclusive_kernel"]) --> B["tid = threadIdx.x"]
    B --> C["gid = blockIdx.x * blockDim.x + tid"]
    C --> D["x = 0"]
    D --> E{"gid < N ?"}
    E -- "Да" --> F["x = in[gid]"]
    E -- "Нет" --> G["x = 0"]
    F --> H["sh[tid] = x"]
    G --> H
    H --> I["__syncthreads()"]

    I --> J["offset = 1"]
    J --> K{"offset < blockDim.x ?"}
    K -- "Да" --> L["val = 0"]
    L --> M{"tid >= offset ?"}
    M -- "Да" --> N["val = sh[tid - offset]"]
    M -- "Нет" --> O["val = 0"]
    N --> P["__syncthreads()"]
    O --> P
    P --> Q["sh[tid] = sh[tid] + val"]
    Q --> R["__syncthreads()"]
    R --> S["offset = offset * 2"]
    S --> K

    K -- "Нет" --> T["inclusive = sh[tid]"]
    T --> U["exclusive = inclusive - x"]
    U --> V{"gid < N ?"}
    V -- "Да" --> W["out[gid] = exclusive"]
    V -- "Нет" --> X["пропуск"]

    W --> Y{"tid == blockDim.x - 1 ?"}
    X --> Y
    Y -- "Да" --> Z["block_sums[blockIdx.x] = sh[tid]"]
    Y -- "Нет" --> AA["пропуск"]
    Z --> AB(["End: scan block"])
    AA --> AB
