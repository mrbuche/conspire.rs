use crate::math::{
    Derivative, Differentiate, Quantity, Tensor, TensorVec,
    integrate::{IntegrationError, Times},
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, SecondOrderOptimization,
        ZerothOrderRootFinding,
    },
    sparse::SparseSolver,
};
use crate::units::Time;

pub(super) mod explicit;
// pub mod implicit;

/// Integrators for explicit differential-algebraic equations using zeroth-order root-finding.
pub trait ExplicitDaeZerothOrderRoot<G, Y, Z, U, V, W, T = Time>
where
    Y: Differentiate<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<G, String>,
        solver: impl ZerothOrderRootFinding<G, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for explicit differential-algebraic equations using first-order root-finding.
pub trait ExplicitDaeFirstOrderRoot<F, J, Y, Z, U, V, W, T = Time>
where
    Y: Differentiate<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for explicit differential-algebraic equations using first-order minimization.
pub trait ExplicitDaeFirstOrderMinimize<F, G, Y, Z, U, V, W, T = Time>
where
    Y: Differentiate<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<G, String>,
        solver: impl FirstOrderOptimization<F, G, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for explicit differential-algebraic equations using second-order minimization.
pub trait ExplicitDaeSecondOrderMinimize<F, J, H, Y, Z, U, V, W, T = Time>
where
    Y: Differentiate<T> + Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        evolution: impl FnMut(Quantity<T>, &Y, &Z) -> Result<Derivative<Y, T>, String>,
        function: impl FnMut(Quantity<T>, &Y, &Z) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<J, String>,
        hessian: impl FnMut(Quantity<T>, &Y, &Z) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Z>,
        time: &[Quantity<T>],
        initial_condition: (Y, Z),
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<(Times<T>, U, W, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using zeroth-order root-finding.
pub trait ImplicitDaeZerothOrderRoot<G, Y, U, V, T = Time>
where
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl ZerothOrderRootFinding<G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using first-order root-finding.
pub trait ImplicitDaeFirstOrderRoot<F, J, Y, U, V, T = Time>
where
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using first-order minimization.
pub trait ImplicitDaeFirstOrderMinimize<F, G, Y, U, V, T = Time>
where
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<G, String>,
        solver: impl FirstOrderOptimization<F, G, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}

/// Integrators for implicit differential-algebraic equations using second-order minimization.
pub trait ImplicitDaeSecondOrderMinimize<F, J, H, Y, U, V, T = Time>
where
    Y: Differentiate<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<F, String>,
        jacobian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<J, String>,
        hessian: impl FnMut(Quantity<T>, &Y, &Derivative<Y, T>) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Derivative<Y, T>>,
        time: &[Quantity<T>],
        initial_condition: Y,
        equality_constraint: impl FnMut(Quantity<T>) -> EqualityConstraint,
        sparse: Option<SparseSolver>,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}
