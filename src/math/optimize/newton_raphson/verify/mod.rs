use super::super::{
    super::{Hessian, sparse::SparseSolver},
    OptimizationError,
};
use std::fmt::Debug;

pub(super) fn not_minimum(
    newton_raphson: &(impl Debug + ?Sized),
    variables: usize,
    constraints: usize,
) -> OptimizationError {
    OptimizationError::NotMinimum(
        format!(
            "a stationary point of {variables} variables and {constraints} constraints \
            whose tangent is not positive definite on the constraints' null space"
        ),
        format!("{newton_raphson:?}"),
    )
}

pub(super) fn verified<H>(
    sparse: Option<SparseSolver>,
) -> impl FnMut(H, &[bool], &[usize]) -> Option<bool>
where
    H: Hessian,
{
    move |hessian, retained, unmap| match &sparse {
        Some(solver) => solver.is_minimum(|i, j| hessian.entry(unmap[i], unmap[j]), 0),
        None => hessian
            .retain_from(retained)
            .factorize_ldl()
            .ok()
            .map(|ldl| ldl.is_minimum(0)),
    }
}
