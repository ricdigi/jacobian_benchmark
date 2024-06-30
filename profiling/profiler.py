import time
import json

from profiling.utils import clear_sympy_cache, warm_up_function
from sympy.physics.mechanics.models import n_link_pendulum_on_cart
from sympy import ImmutableDenseMatrix, Symbol
from sympy.physics.mechanics import dynamicsymbols

from implementations.forward_jacobian_ric import forward_jacobian_ric
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


def save_results_to_json(data, filename='data/results.json'):
    """
    Save profiling results to a JSON file.
    """

    try:
        with open(filename, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results.append(data)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


def run_profiling(num_runs=10, sizes=tuple(range(1, 5))):
    """
    Profile different Jacobian implementations using the given number of runs and input sizes.
    """

    implementations = {
        'jacobian_classic': jacobian_classic,
        'forward_jacobian_ric': forward_jacobian_ric,
        'forward_jacobian_sam': forward_jacobian_sam,
        'jacobian_protosym': jacobian_protosym,
        'jacobian_symengine': jacobian_symengine
    }

    for size in sizes:
        expr, wrt = generate_input(size)

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


def generate_input(n):
    """
    Use the n_link_pendulum_on_cart model from sympy as an example system to
    generate a set of equations and variables for testing the Jacobian implementations
    """

    sys_kane = n_link_pendulum_on_cart(n, cart_force=True, joint_torques=False)

    # Extract coordinates (generalized coordinates)
    coordinates = sys_kane.q
    speeds = sys_kane.u
    equations = sys_kane.kanes_equations()

    # Convert equations to an immutable dense matrix
    expr = ImmutableDenseMatrix(equations[1])

    # Extract free symbols excluding time
    wrt_2 = ImmutableDenseMatrix([[sym] for sym in (equations[1].free_symbols - {dynamicsymbols._t})])
    wrt = ImmutableDenseMatrix([*speeds, *coordinates])

    # Substitute dynamicsymbols with regular symbols for consistency
    new_symbols = {symbol: Symbol(f'f{i + 1}') for i, symbol in enumerate(wrt)}
    expr = expr.subs(new_symbols)
    wrt = wrt.subs(new_symbols)
    wrt = ImmutableDenseMatrix.vstack(wrt, wrt_2)

    return expr, wrt
