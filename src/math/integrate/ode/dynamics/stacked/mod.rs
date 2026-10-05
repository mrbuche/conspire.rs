#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorTuple, TensorVec, TensorVector,
        integrate::{
            BogackiShampine, BogackiShampineFixedStep, DormandPrince, DormandPrinceFixedStep,
            Explicit, ExplicitDynamics, IntegrationError, Spectrum, Times, Verner8,
            Verner8FixedStep, Verner9, Verner9FixedStep,
        },
    },
    units::Time,
};
use std::fmt::Debug;

type Stack<X, T> = TensorTuple<X, Derivative<X, T>>;
type Stacks<X, T> = TensorVector<Stack<X, T>>;
type Slopes<X, T> = TensorVector<Derivative<Stack<X, T>, T>>;

#[doc = include_str!("doc.md")]
pub trait Stacked<T = Time>
where
    Self: Debug,
{
    #[doc = include_str!("../doc.md")]
    fn integrate_stacked<X, UX, UV, UA>(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>
    where
        Self: Explicit<Stack<X, T>, Stacks<X, T>, Slopes<X, T>, T>,
        X: Differentiable<T> + Tensor,
        Derivative<X, T>: Differentiable<T> + Tensor,
        UX: TensorVec<Item = X>,
        UV: TensorVec<Item = Derivative<X, T>>,
        UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    {
        let (times, stacks, slopes) = self.integrate(
            |t, stack: &Stack<X, T>| {
                let acceleration = function(t, &stack.0, &stack.1)?;
                Ok(TensorTuple(stack.1.clone(), acceleration))
            },
            time,
            TensorTuple(initial_position, initial_velocity),
        )?;
        Ok(unstack(times, &stacks, &slopes))
    }
    #[doc = include_str!("../bounded.md")]
    fn integrate_stacked_bounded<X, UX, UV, UA>(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        mut bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<Quantity<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError>
    where
        Self: Explicit<Stack<X, T>, Stacks<X, T>, Slopes<X, T>, T>,
        X: Differentiable<T> + Tensor,
        Derivative<X, T>: Differentiable<T> + Tensor,
        UX: TensorVec<Item = X>,
        UV: TensorVec<Item = Derivative<X, T>>,
        UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    {
        let (times, stacks, slopes) = self.integrate_bounded(
            |t, stack: &Stack<X, T>| {
                let acceleration = function(t, &stack.0, &stack.1)?;
                Ok(TensorTuple(stack.1.clone(), acceleration))
            },
            |t, stack: &Stack<X, T>| Ok(Spectrum::Imaginary(bound(t, &stack.0, &stack.1)?)),
            safety,
            time,
            TensorTuple(initial_position, initial_velocity),
        )?;
        Ok(unstack(times, &stacks, &slopes))
    }
}

fn unstack<X, UX, UV, UA, T>(
    times: Times<T>,
    stacks: &Stacks<X, T>,
    slopes: &Slopes<X, T>,
) -> (Times<T>, UX, UV, UA)
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    let mut positions = UX::new();
    let mut velocities = UV::new();
    let mut accelerations = UA::new();
    stacks.iter().zip(slopes.iter()).for_each(|(stack, slope)| {
        positions.push(stack.0.clone());
        velocities.push(stack.1.clone());
        accelerations.push(slope.1.clone());
    });
    (times, positions, velocities, accelerations)
}

impl<I, X, UX, UV, UA, T> ExplicitDynamics<X, UX, UV, UA, T> for I
where
    I: Stacked<T> + Explicit<Stack<X, T>, Stacks<X, T>, Slopes<X, T>, T>,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
{
    fn integrate(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError> {
        self.integrate_stacked(function, time, initial_position, initial_velocity)
    }
    fn integrate_bounded(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
        ) -> Result<Derivative<Derivative<X, T>, T>, String>,
        bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>) -> Result<Quantity<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
    ) -> Result<(Times<T>, UX, UV, UA), IntegrationError> {
        self.integrate_stacked_bounded(
            function,
            bound,
            safety,
            time,
            initial_position,
            initial_velocity,
        )
    }
}

macro_rules! stacked {
    ($($integrator: ty),+) => {
        $(impl<T> Stacked<T> for $integrator {})+
    };
}

stacked!(
    BogackiShampine,
    BogackiShampineFixedStep,
    DormandPrince,
    DormandPrinceFixedStep,
    Verner8,
    Verner8FixedStep,
    Verner9,
    Verner9FixedStep
);
