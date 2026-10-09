use crate::math::{
    QuantitySparseVec2D, Tensor, Vector,
    integrate::IntegrationError,
    optimize::EqualityConstraint,
    sparse::{CscLdl, CscMatrix, SparseError},
};

/// The indices of the fixed degrees of freedom of a constraint, for an integrator named
/// `integrator`, which does not support linear constraints.
pub(crate) fn fixed_indices(
    equality_constraint: EqualityConstraint,
    integrator: &str,
) -> Result<Vec<usize>, IntegrationError> {
    match equality_constraint {
        EqualityConstraint::Fixed(indices) => Ok(indices),
        EqualityConstraint::None => Ok(vec![]),
        EqualityConstraint::Linear(..) => Err(IntegrationError::Intermediate(format!(
            "Linear constraints are not supported by {integrator}."
        ))),
    }
}

/// The factors of a symmetric positive definite matrix restricted to the degrees of freedom
/// that are not fixed, `D` per node and indexed as `D * node + component`.
pub struct FreeFactored<const D: usize> {
    factors: CscLdl,
    free: Vec<usize>,
}

impl<const D: usize> FreeFactored<D> {
    pub(crate) fn factor<U>(
        matrix: &QuantitySparseVec2D<U>,
        fixed: &[usize],
    ) -> Result<Self, SparseError> {
        let (factors, free) = factor_free::<U, D>(matrix, fixed)?;
        Ok(Self { factors, free })
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

fn factor_free<U, const D: usize>(
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
