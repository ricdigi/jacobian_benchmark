from benchmark.benchmark import run_benchmark_pendulum
from benchmark.benchmark import run_benchmark_bicycle

run_benchmark_pendulum(num_runs=1, sizes=tuple(range(1, 11)))
run_benchmark_bicycle(1)

