use crate::math::{
    Derivative, Differentiate, Quantity, Scalar, Tensor, TensorVec,
    integrate::{ExplicitDaeVariableStepExplicit, ode::explicit::variable_step::verner_9::*},
};
use std::ops::{Mul, Sub};

impl<Y, Z, U, V, W, T> ExplicitDaeVariableStepExplicit<Y, Z, U, V, W, T> for Verner9
where
    Y: Differentiate<T> + Tensor,
    Z: PartialEq + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Z>,
    W: TensorVec<Item = Derivative<Y, T>>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
{
}
