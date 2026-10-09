use crate::math::{
    Derivative, Differentiable, Quantity, Tensor, TensorVec,
    integrate::{IntegrationError, Times},
    optimize::{EqualityConstraint, Optimization, RootFinding},
    sparse::SparseSolver,
};
use crate::units::Time;

pub(super) mod explicit;
// pub mod implicit;

/// Integrators for explicit differential-algebraic equations using root-finding.
pub trait ExplicitDaeRoot<F, J, Y, Z, U, V, W, T = Time>
where
    Y: Differentiable<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<J, String>,
        solver: impl RootFinding<F, J, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for explicit differential-algebraic equations using minimization.
pub trait ExplicitDaeMinimize<F, J, H, Y, Z, U, V, W, T = Time>
where
    Y: Differentiable<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<J, String>,
        hessian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<H, String>,
        solver: impl Optimization<F, J, H, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using root-finding.
pub trait ImplicitDaeRoot<F, J, Y, U, V, T = Time>
where
    Y: Differentiable<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        solver: impl RootFinding<F, J, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using minimization.
pub trait ImplicitDaeMinimize<F, J, H, Y, U, V, T = Time>
where
    Y: Differentiable<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    #[expect(clippy::too_many_arguments)]
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        hessian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<H, String>,
        solver: impl Optimization<F, J, H, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}
