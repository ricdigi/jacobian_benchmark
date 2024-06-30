
import protosym.simplecas as pcas
from sympy import Symbol, Derivative

def jacobian_protosym(M, wrt):
    """
    Implementation proposed by @oscarbenjamin in sympy issue #26730
    """
    # Mask off derivatives
    rep = {d: Symbol(str(d)) for d in M.atoms(Derivative)}
    rep_reverse = {v: k for k, v in rep.items()}

    M = M.xreplace(rep)

    Mp = pcas.Matrix.from_sympy(M.T)
    syms = [pcas.Expr.from_sympy(s) for s in wrt]

    Jp = _jacobian_protosym(Mp, syms)

    # This is the slow part because it creates SymPy expressions:
    J = Jp.to_sympy().T
    J = J.xreplace(rep_reverse)

    return J


def _jacobian_protosym(Mp, syms):
    # protosym Matrix does not have a jacobian method so just
    # diff wrt each symbol and combine
    rows = []
    for sp in syms:
        rows.append(Mp.diff(sp).tolist()[0])
    return pcas.Matrix(rows)

