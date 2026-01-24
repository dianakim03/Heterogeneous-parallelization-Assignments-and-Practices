# Practice 7 — CUDA Reduction & Prefix Sum (Scan)

## Цель
Выполнить три пункта:
1) Реализовать редукцию (сумма массива) на GPU с использованием CUDA и shared memory  
2) Реализовать префиксную сумму (exclusive scan) на GPU с использованием CUDA и shared memory  
3) Замерить время CPU vs GPU для массивов разного размера и построить графики

---

## Что реализовано

### 1) Reduction (sum)
- CUDA kernel `reduce_kernel` загружает элементы в shared memory
- внутри блока выполняется редукция: на каждом шаге активных потоков становится в 2 раза меньше
- на выходе получается массив `block_sums`, где каждый элемент — сумма одного блока
- итоговая сумма на CPU получается сложением `block_sums`

### 2) Exclusive Scan (prefix sum)
- CUDA kernel `scan_block_exclusive_kernel` делает scan внутри каждого блока в shared memory
- сохраняется сумма каждого блока в `block_sums`
- на CPU вычисляются оффсеты блоков `block_offsets`
- kernel `add_offsets_kernel` прибавляет оффсеты, чтобы получить scan для всего массива

### 3) Performance analysis
- CPU version: последовательная редукция и последовательный exclusive scan
- GPU version: редукция и scan на CUDA
- Время GPU измеряется CUDA events (в ms)
- Время CPU измеряется chrono (в ms)
- Результаты записываются в `results.csv` и строятся графики

---

## Проверка корректности
- Для редукции проверяется совпадение сумм CPU и GPU (`sum_ok`)
- Для scan проверяется совпадение первых 100 элементов CPU и GPU (`scan_head100_ok`)

---

## Запуск в Google Colab (Tesla T4)
### Компиляция и запуск
```bash
nvcc -O2 -gencode arch=compute_75,code=sm_75 practice7.cu -o practice7
./practice7
