use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::IntegrationError,
    optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
};
use std::fmt::Debug;

pub mod explicit;

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

pub trait DaeSolverFirstOrderRoot<F, J, Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate_dae(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Z) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}
