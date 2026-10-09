use crate::math::{
    QuantitySparseVec2D, Tensor,
    sparse::{CscLdl, CscMatrix, SparseError},
};

/// Factors a symmetric positive definite matrix restricted to the degrees of freedom that are
/// not fixed, returning the factors and the indices of the free degrees of freedom.
pub(crate) fn factor_free<U, const D: usize>(
    matrix: &QuantitySparseVec2D<U>,
    fixed: &[usize],
) -> Result<(CscLdl, Vec<usize>), SparseError> {
    let mut is_free = vec![true; D * matrix.len()];
    fixed.iter().for_each(|&index| is_free[index] = false);
    let free: Vec<usize> = (0..D * matrix.len()).filter(|&i| is_free[i]).collect();
    let mut reduced = vec![usize::MAX; D * matrix.len()];
    free.iter()
        .enumerate()
        .for_each(|(k, &index)| reduced[index] = k);
    let mut sparse = CscMatrix::from_pattern(
        free.len(),
        free.len(),
        matrix
            .iter()
            .enumerate()
            .flat_map(|(a, row)| {
                row.entries()
                    .flat_map(move |(b, _)| (0..D).map(move |i| (D * a + i, D * b + i)))
            })
            .filter(|&(row, column)| is_free[row] && is_free[column])
            .map(|(row, column)| (reduced[row], reduced[column]))
            .collect(),
    );
    sparse.fill(|row, column| matrix[free[row] / D][free[column] / D].value());
    let mut factors = sparse.ldl_symbolic()?;
    factors.refactor(&sparse)?;
    Ok((factors, free))
}
