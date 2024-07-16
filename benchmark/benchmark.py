import time
import json

from benchmark.utils import clear_sympy_cache, warm_up_function
from benchmark.models import generate_input_pendulum, generate_input_bicycle

from implementations.forward_jacobian_ric import forward_jacobian_ric
from implementations.forward_jacobian_ric2 import forward_jacobian_ric2
from implementations.forward_jacobian_ric3 import forward_jacobian_ric3
from implementations.forward_jacobian_ric4 import forward_jacobian_ric4
from implementations.forward_jacobian_sam import forward_jacobian_sam
from implementations.jacobian_classic import jacobian_classic
from implementations.jacobian_protosym import jacobian_protosym
from implementations.jacobian_symengine import jacobian_symengine


def time_function(func, *args, **kwargs):
    """
    Times the execution of a given function.
    """

    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time, result


def save_results_to_json(data, filename='data/results_pendulum.json'):
    """
    Save benchmark results to a JSON file.
    """

    try:
        with open(filename, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results.append(data)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def run_benchmark_pendulum(num_runs=10, sizes=tuple(range(1, 5))):
    """
    Benchmark different Jacobian implementations using the given number of runs and input sizes.
    """

    implementations = {
        #'jacobian_classic': jacobian_classic,
        #'forward_jacobian_ric': forward_jacobian_ric,
        #'forward_jacobian_ric2': forward_jacobian_ric2,
        #'forward_jacobian_ric3': forward_jacobian_ric3,
        'forward_jacobian_ric4': forward_jacobian_ric4,
        #'forward_jacobian_sam': forward_jacobian_sam,
        #'jacobian_protosym': jacobian_protosym,
        #'jacobian_symengine': jacobian_symengine
    }

    for size in sizes:
        expr, wrt = generate_input_pendulum(size)

        for name, func in implementations.items():
            clear_sympy_cache()
            warm_up_function(func, expr, wrt)  # Warm up the function

            sub_times = {'total': []}
            for _ in range(num_runs):
                clear_sympy_cache()
                total_time, _ = time_function(func, expr, wrt)
                sub_times['total'].append(total_time)

            # Average the results
            avg_total_time = sum(sub_times['total']) / num_runs

            # Save results
            data = {
                'implementation': name,
                'input_size': len(expr),
                'wrt_size': len(wrt),
                'total_time': avg_total_time
            }

            save_results_to_json(data)
            print(f"{name} - Input Size: {len(expr)}, Total Time: {avg_total_time}, Sub Times: {sub_times}")


def run_benchmark_bicycle(num_runs=10):
    """
    Benchmark different Jacobian implementations using the given number of runs and input sizes.
    """

    implementations = {
        #'jacobian_classic': jacobian_classic,
        #'forward_jacobian_ric': forward_jacobian_ric,
        #'forward_jacobian_ric2': forward_jacobian_ric2,
        #'forward_jacobian_ric3': forward_jacobian_ric3,
        'forward_jacobian_ric4': forward_jacobian_ric4,
        #'forward_jacobian_sam': forward_jacobian_sam,
        #'jacobian_protosym': jacobian_protosym,
        #'jacobian_symengine': jacobian_symengine
    }


    expr, wrt = generate_input_bicycle()

    for name, func in implementations.items():
        clear_sympy_cache()
        warm_up_function(func, expr, wrt)  # Warm up the function

        sub_times = {'total': []}
        for _ in range(num_runs):
            clear_sympy_cache()
            total_time, _ = time_function(func, expr, wrt)
            sub_times['total'].append(total_time)

        # Average the results
        avg_total_time = sum(sub_times['total']) / num_runs

        # Save results
        data = {
            'implementation': name,
            'input_size': len(expr),
            'wrt_size': len(wrt),
            'total_time': avg_total_time
        }

        save_results_to_json(data, filename='data/results_bicycle.json')
        print(f"{name} - Input Size: {len(expr)}, Total Time: {avg_total_time}, Sub Times: {sub_times}")
