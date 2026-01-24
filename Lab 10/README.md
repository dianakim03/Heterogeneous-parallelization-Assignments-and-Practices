# Practice 10 — Performance (OpenMP + CUDA + Hybrid + MPI)

## Описание
Практическая работа состоит из 4 заданий:
1) OpenMP: параллельная обработка массива (sum/mean/var/std) + анализ ускорения и закона Амдала  
2) CUDA: сравнение паттернов доступа к памяти (coalesced vs uncoalesced) + оптимизация через shared memory  
3) Hybrid CPU+GPU: асинхронные копии `cudaMemcpyAsync` и CUDA streams + профилирование накладных расходов  
4) MPI: масштабируемость распределённой агрегации (Reduce и Allreduce), strong/weak scaling

Среда: Google Colab, компиляция через `g++/nvcc/mpic++`, запуск через `./program` и `mpirun`.

---

## Task 1 — OpenMP (профилирование + Амдал)
### Что делается
- Создаётся массив `N`
- Считается сумма и сумма квадратов:
  - последовательно (seq)
  - параллельно через OpenMP (omp + reduction)
- Замер времени через `omp_get_wtime()`
- Строится ускорение: `speedup = seq_time / omp_time`
- Оценивается последовательная доля (примерно по Амдалу): `serial_part f` и `parallel_part = 1 - f`

### Запуск (пример)
```bash
g++ -O2 -fopenmp task1_omp.cpp -o task1_omp
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DYNAMIC=false

./task1_omp 10000000 1 5
./task1_omp 10000000 2 5
./task1_omp 10000000 4 5
./task1_omp 10000000 8 5
