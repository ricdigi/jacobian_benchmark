from sympy import cse, Matrix, zeros


def forward_jacobian_ric2(expr, wrt):

    replacements, reduced_expr = cse(expr)

    rep_sym, sub_expr = zip(*replacements)
    rep_sym = Matrix(rep_sym)
    sub_expr = Matrix(sub_expr)

    # Function to perform msparse matrix multiplication in dok format
    def dok_matrix_multiply(A, B):
        result = {}
        for (i, k), A_value in A.items():
            for (k2, j), B_value in B.items():
                if k == k2:
                    if (i, j) in result:
                        result[(i, j)] += A_value * B_value
                    else:
                        result[(i, j)] = A_value * B_value
        return result

    def sparse_matrix_multiply_and_update(B, C, num_rows_B):
        result = {}

        for i in range(num_rows_B):
            # Extract the relevant elements of B for row i
            row_elements_B = {k: v for k, v in B.items() if k[0] == i}

            for (i, j), B_ij in row_elements_B.items():
                for (j_k, k), C_jk in C.items():
                    if j_k == j:
                        if (i, k) in result:
                            result[(i, k)] += B_ij * C_jk
                        else:
                            result[(i, k)] = B_ij * C_jk

            # Add the results of the current row to C and reinitialize the result dictionary
            for key, value in result.items():
                if key in C:
                    C[key] += value
                else:
                    C[key] = value

            result.clear()

    # Computing Sparse Matrix A
    A_sparse = {
        (i, j): diff_value
        for i, s in enumerate(sub_expr)
        for j, w in enumerate(wrt)
        if (diff_value := s.diff(w)) != 0
    }

    # Computing Sparse Matrix B
    precomputed_fs = [s.free_symbols for s in sub_expr]
    B_sparse = {
        (i, j): diff_value
        for i in range(len(sub_expr))
        for j in range(i + 1)
        if rep_sym[j] in precomputed_fs[i] and (diff_value := sub_expr[i].diff(rep_sym[j])) != 0
    }

    if B_sparse:
        num_B = list(B_sparse.keys())[-1][0]
    else:
        num_B = 0

    # Computing Sparse Matrix C
    C_sparse = A_sparse.copy()
    sparse_matrix_multiply_and_update(B_sparse, C_sparse, num_B)

    # Precompute gradients f_1 and f_2 using Jacobian for sparse matrices
    f1_sparse = {
        (i, j): diff_value
        for i, r in enumerate(reduced_expr[0])
        for j, w in enumerate(wrt)
        if (diff_value := r.diff(w)) != 0
    }

    f2_sparse = {
        (i, j): diff_value
        for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
        for j, s in enumerate(rep_sym)
        if s in fs and (diff_value := r.diff(s)) != 0
    }

    # Compute the final Jacobian matrix J for sparse matrices
    J_sparse = f1_sparse.copy()
    temp_J_sparse = dok_matrix_multiply(f2_sparse, C_sparse)
    for (i, j), value in temp_J_sparse.items():
        if (i, j) in J_sparse:
            J_sparse[(i, j)] += value
        else:
            J_sparse[(i, j)] = value

    # Timing the creation of the substitution dictionary
    sub_rep = {rep_sym: sub_expr for rep_sym, sub_expr in replacements}

    # Perform substitutions in reverse order
    result = zeros(len(expr), len(wrt))
    for (i, j), value in J_sparse.items():
        result[i, j] = value

    sub_dict = {}
    for i in range(0, num_B):
        interest_keys = [key[1] for key in B_sparse.keys() if key[0] == i]
        for j in interest_keys:
            sub_dict[rep_sym[j]] = sub_rep[rep_sym[j]]
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)
        sub_dict = {}

    result = result.xreplace(sub_rep)

    return result  # , A_sparse, B_sparse, C_sparse, f1_sparse, f2_sparse