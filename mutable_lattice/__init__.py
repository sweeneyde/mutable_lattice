from _mutable_lattice import (
    Vector,
    row_op,
    generalized_row_op,
    Lattice,
    xgcd,
    relations_among_c,
    decompose_relations_among,
    transpose,
)

def relations_among_with_decomposition(vecs, /):
    R = len(vecs)
    relations, subproblem_rows, subproblems = decompose_relations_among(vecs)
    for rows, subproblem in zip(subproblem_rows, subproblems, strict=True):
        for rel in relations_among_c(subproblem).get_basis():
            relations.append(rel.shuffled_by_action(rows, R))
    return Lattice(R, relations, maxrank=len(relations))

relations_among = relations_among_with_decomposition
