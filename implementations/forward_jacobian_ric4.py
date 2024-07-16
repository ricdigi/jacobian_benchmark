from sympy import cse, Matrix, SparseMatrix, Derivative
from collections import Counter


def postprocess(repl, reduced):
    """
    Postprocess the CSE output to remove any CSE replacement symbols from the arguments of Derivative terms.
    """

    repl_dict = dict(repl)

    p_repl = [(rep_sym, traverse(sub_exp, repl_dict)) for rep_sym, sub_exp in repl]
    p_reduced = [traverse(red_exp, repl_dict) for red_exp in reduced]

    return p_repl, p_reduced


def traverse(node, repl_dict):
    """
    Traverse the node in preorder fashion, and apply replace_all() if the node
    is the argument of a Derivative.
    """

    if isinstance(node, Derivative):
        return replace_all(node, repl_dict)

    if not node.args:
        return node

    new_args = [traverse(arg, repl_dict) for arg in node.args]
    return node.func(*new_args)


def replace_all(node, repl_dict):
    """
    Bring the node to its form before the CSE operation, by iteratively substituting
    the CSE replacement symbols in the node.
    """

    result = node
    while True:
        fs = result.free_symbols
        sl_dict = {k: repl_dict[k] for k in fs if k in repl_dict}
        if not sl_dict:
            break
        result = result.xreplace(sl_dict)
    return result


def dok_matrix_multiply(A, B):
    """
    Multiply two sparse matrices in dok format (i, k) and (k, j) to get a dictionary of keys (i, j).
    """
    result = Counter()
    for (i, k), A_value in A.items():
        for (k2, j), B_value in B.items():
            if k == k2:
                result[(i, j)] += A_value * B_value
    return result


def forward_jacobian_ric4(expr, wrt):

    replacements, reduced_expr = cse(expr)
    repl_dict = dict(replacements)

    replacements, reduced_expr = postprocess(replacements, reduced_expr, repl_dict)

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
