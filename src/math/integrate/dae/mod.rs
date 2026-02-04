use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::IntegrationError,
    optimize::{EqualityConstraint, ZerothOrderRootFinding},
};
use std::fmt::Debug;

pub mod explicit;

/// Differential-algebraic equation integrators.
pub trait DaeSolver<Y, Z, U, V>
where
    Self: Debug,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
}

pub trait DaeSolverZerothOrderRoot<Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    fn integrate_dae(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        solver: impl ZerothOrderRootFinding<Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}
