from sympy import cse, Matrix, SparseMatrix
from sympy.polys.matrices.sdm import SDM, sdm_matmul_exraw
from sympy import EXRAW

def forward_jacobian_sdm(expr, wrt):
    # CSE
    replacements, reduced_expr = cse(expr)
    rep_sym, sub_expr = map(Matrix, zip(*replacements))
    l_sub, l_wrt, l_red = len(sub_expr), len(wrt), len(reduced_expr[0])

    f1 = SDM.from_dok({(i, j): r.diff(w) for i, r in enumerate(reduced_expr[0])
                       for j, w in enumerate(wrt) if r.diff(w) != 0}, (l_red, l_wrt), EXRAW)

    f2 = SDM.from_dok({(i, j): r.diff(s) for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
                       for j, s in enumerate(rep_sym) if s in fs and r.diff(s) != 0}, (l_red, l_sub), EXRAW)

    Ai = {(0, j): diff_value for j, w in enumerate(wrt) if (diff_value := sub_expr[0].diff(w)) != 0}
    C = SDM.from_dok(Ai, (1, l_wrt), EXRAW)

    symbols = expr.free_symbols
    precomputed_fs = [s.free_symbols - symbols for s in sub_expr]

    for i in range(1, l_sub):

        Bi = SDM.from_dok({(0, j): sub_expr[i].diff(rep_sym[j]) for j in range(i)
                           if rep_sym[j] in precomputed_fs[i] and sub_expr[i].diff(rep_sym[j]) != 0}, (1, i), EXRAW)

        Ai = SDM.from_dok({(0, j): sub_expr[i].diff(w) for j, w in enumerate(wrt)
                           if sub_expr[i].diff(w) != 0}, (1, l_wrt), EXRAW)

        if Bi :
            Ci = sdm_matmul_exraw(Bi, C, Bi.domain, 1, l_wrt)
            Ci = Bi.new(Ci, (1, l_wrt), Bi.domain).add(Ai)
        else:
            Ci = Ai

        C.shape = (i + 1, l_wrt)
        if Ci: C[i] = Ci[0]

    # Differentiate step
    Jsdm = sdm_matmul_exraw(f2, C, f2.domain, f2.shape[0], C.shape[1])
    Jsdm = f2.new(Jsdm, (f2.shape[0], C.shape[1]), f2.domain).add(f1)

    sub_rep = {rep_sym: sub_expr for rep_sym, sub_expr in replacements}
    for i, ik in enumerate(precomputed_fs):
        sub_dict = {j: sub_rep[j] for j in ik}
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)

    Jdok = {key: expr.xreplace(sub_rep) for key, expr in Jsdm.to_dok().items()}
    J = SparseMatrix(None, Jdok)

    return J

