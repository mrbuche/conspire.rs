use crate::math::{
    Scalar, Tensor, TensorVec, Vector,
    integrate::{Explicit, IntegrationError},
};

/// Explicit ordinary differential equation solvers with internal variables.
pub trait ExplicitInternalVariables<Y, Z, U, V>
where
    Self: Explicit<Y, U>,
    Y: Tensor,
    Z: Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
{
    #[doc = include_str!("doc.md")]
    fn integrate_and_evaluate(
        &self,
        function: impl FnMut(Scalar, &Y, &Z) -> Result<Y, String>,
        evaluate: impl FnMut(Scalar, &Y, &Z) -> Result<Z, String>,
        time: &[Scalar],
        initial_condition: Y,
        initial_evaluation: Z,
    ) -> Result<(Vector, U, U, V), IntegrationError>;
}
