use crate::math::{
    Banded, Scalar, Tensor, TensorVec, Vector,
    integrate::IntegrationError,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, SecondOrderOptimization,
        ZerothOrderRootFinding,
    },
};

pub mod explicit;
// pub mod implicit;

pub trait ExplicitDaeZerothOrderRoot<Y, Z, U, V>
where
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

pub trait ExplicitDaeFirstOrderRoot<F, J, Y, Z, U, V>
where
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

pub trait ExplicitDaeFirstOrderMinimize<F, Y, Z, U, V>
where
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

pub trait ExplicitDaeSecondOrderMinimize<F, J, H, Y, Z, U, V>
where
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

pub trait ImplicitDaeZerothOrderRoot<Y, U>
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y, &Y) -> Result<Y, String>,
        solver: impl ZerothOrderRootFinding<Y>,
        time: &[Scalar],
        initial_condition: Y,
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

pub trait ImplicitDaeFirstOrderRoot<F, J, Y, U>
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y, &Y) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Y) -> Result<J, String>,
        solver: impl FirstOrderRootFinding<F, J, Y>,
        time: &[Scalar],
        initial_condition: Y,
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

pub trait ImplicitDaeFirstOrderMinimize<F, Y, U>
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y, &Y) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Y) -> Result<Y, String>,
        solver: impl FirstOrderOptimization<F, Y>,
        time: &[Scalar],
        initial_condition: Y,
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
    ) -> Result<(Vector, U, U), IntegrationError>;
}

pub trait ImplicitDaeSecondOrderMinimize<F, J, H, Y, U>
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        function: impl FnMut(Scalar, &Y, &Y) -> Result<F, String>,
        jacobian: impl FnMut(Scalar, &Y, &Y) -> Result<J, String>,
        hessian: impl FnMut(Scalar, &Y, &Y) -> Result<H, String>,
        solver: impl SecondOrderOptimization<F, J, H, Y>,
        time: &[Scalar],
        initial_condition: Y,
        equality_constraint: impl FnMut(Scalar) -> EqualityConstraint,
        banded: Option<Banded>,
    ) -> Result<(Vector, U, U), IntegrationError>;
}
