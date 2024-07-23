from sympy import cse, Matrix, SparseMatrix, Function
from collections import Counter


def dok_matrix_multiply(A, B):
    result = Counter()
    for (i, k), A_value in A.items():
        for (k2, j), B_value in B.items():
            if k == k2:
                result[(i, j)] += A_value * B_value
    return result


def forward_jacobian_ric3(expr, wrt):
    # Have both wrt and expr in the cse operation
    concatenated = Matrix.vstack(expr, wrt)
    replacements, reduced_concatenated = cse(concatenated)

    reduced_expr = [reduced_concatenated[0][:expr.shape[0], :]]
    wrt = reduced_concatenated[0][expr.shape[0]:, :]

    if replacements:
        rep_sym, sub_expr = map(Matrix, zip(*replacements))
    else:
        rep_sym, sub_expr, reduced_expr = Matrix([]), Matrix([]), [expr]

    l_sub, l_wrt, l_red = len(sub_expr), len(wrt), len(reduced_expr[0])

    f1 = {
        (i, j): diff_value
        for i, r in enumerate(reduced_expr[0])
        for j, w in enumerate(wrt)
        if (diff_value := r.diff(w)) != 0
    }

    if not replacements:
        return SparseMatrix(l_red, l_wrt, f1)

    f2 = {
        (i, j): diff_value
        for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
        for j, s in enumerate(rep_sym)
        if s in fs and (diff_value := r.diff(s)) != 0
    }

    symbols = expr.free_symbols
    precomputed_fs = [s.free_symbols - symbols for s in sub_expr]

    C = Counter({(0, j): diff_value for j, w in enumerate(wrt) if (diff_value := sub_expr[0].diff(w)) != 0})

    for i in range(1, l_sub):
        Bi = {(i, j): diff_value for j in range(i + 1)
              if rep_sym[j] in precomputed_fs[i] and (diff_value := sub_expr[i].diff(rep_sym[j])) != 0}

        Ai = Counter({(i, j): diff_value for j, w in enumerate(wrt)
                      if (diff_value := sub_expr[i].diff(w)) != 0})

        if Bi:
            Ci = dok_matrix_multiply(Bi, C)
            Ci.update(Ai)  # Use Counter's update method to add Ai to Ci
            C.update(Ci)  # Update C with the result
        else:
            C.update(Ai)

    J = dok_matrix_multiply(f2, C)

    for (i, j), value in f1.items():
        J[(i, j)] += value

    sub_rep = dict(replacements)
    for i, ik in enumerate(precomputed_fs):
        sub_dict = {j: sub_rep[j] for j in ik}
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)

    J = {key: expr.xreplace(sub_rep) for key, expr in J.items()}
    J = SparseMatrix(None, J)

    return J
