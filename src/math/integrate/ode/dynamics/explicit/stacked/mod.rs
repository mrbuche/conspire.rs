#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorTuple, TensorVec, TensorVector,
        integrate::{
            BogackiShampine, BogackiShampineFixedStep, DormandPrince, DormandPrinceFixedStep,
            Explicit, ExplicitDynamics, ExplicitDynamicsCoupled, IntegrationError, Spectrum, Times,
            Verner8, Verner8FixedStep, Verner9, Verner9FixedStep,
        },
    },
    units::Time,
};
use std::fmt::Debug;

type Stack<X, T> = TensorTuple<X, Derivative<X, T>>;
type Stacks<X, T> = TensorVector<Stack<X, T>>;
type Slopes<X, T> = TensorVector<Derivative<Stack<X, T>, T>>;
type Coupled<X, Y, T> = TensorTuple<Stack<X, T>, Y>;
type Coupleds<X, Y, T> = TensorVector<Coupled<X, Y, T>>;
type CoupledSlopes<X, Y, T> = TensorVector<Derivative<Coupled<X, Y, T>, T>>;

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
    #[doc = include_str!("bounded.md")]
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
    #[expect(clippy::type_complexity)]
    fn integrate_stacked_coupled<X, Y, UX, UV, UA, UY, UR>(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError>
    where
        Self: Explicit<Coupled<X, Y, T>, Coupleds<X, Y, T>, CoupledSlopes<X, Y, T>, T>,
        X: Differentiable<T> + Tensor,
        Derivative<X, T>: Differentiable<T> + Tensor,
        Y: Differentiable<T> + Tensor,
        UX: TensorVec<Item = X>,
        UV: TensorVec<Item = Derivative<X, T>>,
        UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
        UY: TensorVec<Item = Y>,
        UR: TensorVec<Item = Derivative<Y, T>>,
    {
        let (times, coupleds, slopes) = self.integrate(
            |t, coupled: &Coupled<X, Y, T>| {
                let (acceleration, rate) = function(t, &coupled.0.0, &coupled.0.1, &coupled.1)?;
                Ok(TensorTuple(
                    TensorTuple(coupled.0.1.clone(), acceleration),
                    rate,
                ))
            },
            time,
            TensorTuple(
                TensorTuple(initial_position, initial_velocity),
                initial_state,
            ),
        )?;
        Ok(unstack_coupled(times, &coupleds, &slopes))
    }
    #[expect(clippy::type_complexity)]
    #[expect(clippy::too_many_arguments)]
    fn integrate_stacked_coupled_bounded<X, Y, UX, UV, UA, UY, UR>(
        &self,
        mut function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        mut bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError>
    where
        Self: Explicit<Coupled<X, Y, T>, Coupleds<X, Y, T>, CoupledSlopes<X, Y, T>, T>,
        X: Differentiable<T> + Tensor,
        Derivative<X, T>: Differentiable<T> + Tensor,
        Y: Differentiable<T> + Tensor,
        UX: TensorVec<Item = X>,
        UV: TensorVec<Item = Derivative<X, T>>,
        UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
        UY: TensorVec<Item = Y>,
        UR: TensorVec<Item = Derivative<Y, T>>,
    {
        let (times, coupleds, slopes) = self.integrate_bounded(
            |t, coupled: &Coupled<X, Y, T>| {
                let (acceleration, rate) = function(t, &coupled.0.0, &coupled.0.1, &coupled.1)?;
                Ok(TensorTuple(
                    TensorTuple(coupled.0.1.clone(), acceleration),
                    rate,
                ))
            },
            |t, coupled: &Coupled<X, Y, T>| bound(t, &coupled.0.0, &coupled.0.1, &coupled.1),
            safety,
            time,
            TensorTuple(
                TensorTuple(initial_position, initial_velocity),
                initial_state,
            ),
        )?;
        Ok(unstack_coupled(times, &coupleds, &slopes))
    }
}

fn unstack_coupled<X, Y, UX, UV, UA, UY, UR, T>(
    times: Times<T>,
    coupleds: &Coupleds<X, Y, T>,
    slopes: &CoupledSlopes<X, Y, T>,
) -> (Times<T>, UX, UV, UA, UY, UR)
where
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    Y: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    UY: TensorVec<Item = Y>,
    UR: TensorVec<Item = Derivative<Y, T>>,
{
    let mut positions = UX::new();
    let mut velocities = UV::new();
    let mut accelerations = UA::new();
    let mut states = UY::new();
    let mut rates = UR::new();
    coupleds
        .iter()
        .zip(slopes.iter())
        .for_each(|(coupled, slope)| {
            positions.push(coupled.0.0.clone());
            velocities.push(coupled.0.1.clone());
            accelerations.push(slope.0.1.clone());
            states.push(coupled.1.clone());
            rates.push(slope.1.clone());
        });
    (times, positions, velocities, accelerations, states, rates)
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

impl<I, X, Y, UX, UV, UA, UY, UR, T> ExplicitDynamicsCoupled<X, Y, UX, UV, UA, UY, UR, T> for I
where
    I: Stacked<T> + Explicit<Coupled<X, Y, T>, Coupleds<X, Y, T>, CoupledSlopes<X, Y, T>, T>,
    X: Differentiable<T> + Tensor,
    Derivative<X, T>: Differentiable<T> + Tensor,
    Y: Differentiable<T> + Tensor,
    UX: TensorVec<Item = X>,
    UV: TensorVec<Item = Derivative<X, T>>,
    UA: TensorVec<Item = Derivative<Derivative<X, T>, T>>,
    UY: TensorVec<Item = Y>,
    UR: TensorVec<Item = Derivative<Y, T>>,
{
    fn integrate_coupled(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError> {
        self.integrate_stacked_coupled(
            function,
            time,
            initial_position,
            initial_velocity,
            initial_state,
        )
    }
    fn integrate_coupled_bounded(
        &self,
        function: impl FnMut(
            Quantity<T>,
            &X,
            &Derivative<X, T>,
            &Y,
        )
            -> Result<(Derivative<Derivative<X, T>, T>, Derivative<Y, T>), String>,
        bound: impl FnMut(Quantity<T>, &X, &Derivative<X, T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_position: X,
        initial_velocity: Derivative<X, T>,
        initial_state: Y,
    ) -> Result<(Times<T>, UX, UV, UA, UY, UR), IntegrationError> {
        self.integrate_stacked_coupled_bounded(
            function,
            bound,
            safety,
            time,
            initial_position,
            initial_velocity,
            initial_state,
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
