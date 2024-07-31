from benchmark.benchmark import run_benchmark_pendulum
from benchmark.benchmark import run_benchmark_bicycle
from benchmark.benchmark import run_benchmark_linearize


run_benchmark_pendulum(num_runs=5, sizes=tuple(range(1, 12)))
#run_benchmark_bicycle(1)
#run_benchmark_linearize(5)


