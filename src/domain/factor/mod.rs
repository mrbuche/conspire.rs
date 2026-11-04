use crate::math::{
    QuantitySparseVec2D, Scalar, Tensor, TensorRank2SparseVec2D, Vector,
    sparse::{CscLdl, CscMatrix, SparseError},
};

/// The factors of a symmetric positive definite matrix restricted to the degrees of freedom
/// that are not fixed, `D` per node and indexed as `D * node + component`.
pub struct FreeFactored<const D: usize> {
    factors: CscLdl,
    free: Vec<usize>,
}

impl<const D: usize> FreeFactored<D> {
    /// Factors a matrix with one value per pair of nodes, shared by their `D` components.
    pub(crate) fn factor<U>(
        matrix: &QuantitySparseVec2D<U>,
        fixed: &[usize],
    ) -> Result<Self, SparseError> {
        let pattern = matrix.iter().enumerate().flat_map(|(a, row)| {
            row.entries()
                .flat_map(move |(b, _)| (0..D).map(move |i| (D * a + i, D * b + i)))
        });
        let (factors, free) = factor_free::<D>(
            matrix.len(),
            pattern,
            |row, column| matrix[row / D][column / D].value(),
            fixed,
        )?;
        Ok(Self { factors, free })
    }
    /// Factors a matrix of `D` by `D` blocks, one per pair of nodes.
    pub(crate) fn factor_blocks<I, J, U>(
        matrix: &TensorRank2SparseVec2D<D, I, J, U>,
        fixed: &[usize],
    ) -> Result<Self, SparseError> {
        let pattern = matrix.iter().enumerate().flat_map(|(a, row)| {
            row.entries().flat_map(move |(b, _)| {
                (0..D).flat_map(move |i| (0..D).map(move |j| (D * a + i, D * b + j)))
            })
        });
        let (factors, free) = factor_free::<D>(
            matrix.len(),
            pattern,
            |row, column| matrix[row / D][column / D][row % D][column % D].value(),
            fixed,
        )?;
        Ok(Self { factors, free })
    }
    /// The degrees of freedom that are not fixed.
    pub(crate) fn free(&self) -> &[usize] {
        &self.free
    }
    /// Solves for the right-hand side on every degree of freedom, giving no solution on the
    /// fixed ones, and on the others accounting for the fixed ones not changing.
    pub(crate) fn solve(&self, right_hand_side: &Vector) -> Vector {
        let free_right_hand_side: Vector = self
            .free
            .iter()
            .map(|&index| right_hand_side[index])
            .collect();
        let free_solution = self.factors.solve(&free_right_hand_side);
        let mut solution = Vector::zero(right_hand_side.len());
        self.free
            .iter()
            .enumerate()
            .for_each(|(k, &index)| solution[index] = free_solution[k]);
        solution
    }
}

fn factor_free<const D: usize>(
    nodes: usize,
    pattern: impl Iterator<Item = (usize, usize)>,
    value: impl Fn(usize, usize) -> Scalar,
    fixed: &[usize],
) -> Result<(CscLdl, Vec<usize>), SparseError> {
    let mut is_free = vec![true; D * nodes];
    fixed.iter().for_each(|&index| is_free[index] = false);
    let free: Vec<usize> = (0..D * nodes).filter(|&i| is_free[i]).collect();
    let mut reduced = vec![usize::MAX; D * nodes];
    free.iter()
        .enumerate()
        .for_each(|(k, &index)| reduced[index] = k);
    let mut sparse = CscMatrix::from_pattern(
        free.len(),
        free.len(),
        pattern
            .filter(|&(row, column)| is_free[row] && is_free[column])
            .map(|(row, column)| (reduced[row], reduced[column]))
            .collect(),
    );
    sparse.fill(|row, column| value(free[row], free[column]));
    let mut factors = sparse.ldl_symbolic()?;
    factors.refactor(&sparse)?;
    Ok((factors, free))
}
