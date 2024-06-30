import pytest
from sympy import Matrix, ImmutableDenseMatrix, Symbol, simplify
from sympy.physics.mechanics.models import n_link_pendulum_on_cart
from sympy.physics.mechanics import dynamicsymbols

from implementations.forward_jacobian_ric import forward_jacobian_ric
from implementations.forward_jacobian_sam import forward_jacobian_sam
from implementations.jacobian_classic import jacobian_classic
from implementations.jacobian_protosym import jacobian_protosym
from implementations.jacobian_symengine import jacobian_symengine


@pytest.fixture
def setup_inputs(n = 4):
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


def test_forward_jacobian_ric(setup_inputs):
    expr, wrt = setup_inputs

    # Compute the Jacobian using each implementation
    jacobian_ric = forward_jacobian_ric(expr, wrt)
    jacobian_cla = jacobian_classic(expr, wrt)

    diff = simplify(jacobian_ric - jacobian_cla)

    print(diff)

    # Check that all Jacobians are the same
    assert diff == Matrix.zeros(*diff.shape)


def test_forward_jacobian_sam(setup_inputs):
    expr, wrt = setup_inputs

    # Compute the Jacobian using each implementation
    jacobian_sam = forward_jacobian_sam(expr, wrt)
    jacobian_cla = jacobian_classic(expr, wrt)

    diff = simplify(jacobian_sam - jacobian_cla)

    print(diff)

    # Check that all Jacobians are the same
    assert diff == Matrix.zeros(*diff.shape)


def test_jacobian_protosym(setup_inputs):
    expr, wrt = setup_inputs

    # Compute the Jacobian using each implementation
    jacobian_pro = jacobian_protosym(expr, wrt)
    jacobian_cla = jacobian_classic(expr, wrt)

    diff = simplify(jacobian_pro - jacobian_cla)

    print(diff)

    # Check that all Jacobians are the same
    assert diff == Matrix.zeros(*diff.shape)


def test_jacobian_symengine(setup_inputs):
    expr, wrt = setup_inputs

    # Compute the Jacobian using each implementation
    jacobian_sym = jacobian_symengine(expr, wrt)
    jacobian_cla = jacobian_classic(expr, wrt)

    diff = simplify(jacobian_sym - jacobian_cla)

    print(diff)

    # Check that all Jacobians are the same
    assert diff == Matrix.zeros(*diff.shape)


