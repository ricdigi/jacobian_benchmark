import symengine
from sympy import Matrix

def jacobian_symengine(expr, wrt):
    """
    Implementation proposed by @moorepants in sympy issue #26730
    """
    symeng_expr = symengine.sympify(expr)
    symeng_wrt = symengine.sympify(wrt)

    jac = []

    for e in symeng_expr:
        row = []
        for v in symeng_wrt:
            row.append(e.diff(v))
        jac.append(row)

    # Convert the result to a sympy.Matrix
    jac_matrix = Matrix(jac)

    return jac_matrix
