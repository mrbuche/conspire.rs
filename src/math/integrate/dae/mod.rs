use crate::math::{
    Banded, Scalar, Tensor, TensorVec, Vector,
    integrate::IntegrationError,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, SecondOrderOptimization,
        ZerothOrderRootFinding,
    },
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
    fn integrate(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        solver: impl ZerothOrderRootFinding<Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}

pub trait DaeSimpleZerothOrderRoot<Y, U>
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y, &Y) -> Result<Y, String>,
        solver: impl ZerothOrderRootFinding<Y>,
        time: &[Scalar],
        initial_condition: (Y, Y),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U), IntegrationError>;
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
    fn integrate(
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

pub trait DaeSolverFirstOrderMinimize<F, Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        solver: impl FirstOrderOptimization<F, Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}

pub trait DaeSolverSecondOrderMinimize<F, J, H, Y, Z, U, V>
where
    Self: DaeSolver<Y, Z, U, V>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Z) -> Result<J, String>,
        hessian: impl FnMut(Scalar, &Y, &Z) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Z>,
        time: &[Scalar],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}
