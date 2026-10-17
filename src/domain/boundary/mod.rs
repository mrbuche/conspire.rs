//! Boundary conditions on the degrees of freedom of a model.

mod condition;

pub use condition::{
    BoundaryCondition, BoundaryConditions, Constrain, ConstrainAt, Enforcement, Timed,
};

use crate::{
    math::{
        Current, Quantity, Tensor, TensorRank1, TensorTuple, TensorVector, optimize::Prescription,
    },
    units::{
        Concentration as ConcentrationUnit, ElectricPotential, Length,
        Temperature as TemperatureUnit, Velocity as VelocityUnit,
    },
};
use std::marker::PhantomData;

/// How the degrees of freedom are laid out, `STRIDE` to a node, indexed as `STRIDE * node + offset`.
pub trait Layout {
    const STRIDE: usize;
}

/// The state of a node, whose degrees of freedom are laid out in a way that is a function of it.
pub trait NodalDof {
    type Layout: Layout;
}

/// The state of the nodes, whose degrees of freedom are laid out in a way that is a function of it.
pub trait NodalState {
    type Layout: Layout;
}

/// The coordinates of a solid in `D` dimensions.
pub struct Solid<const D: usize>;

/// The velocities of a solid in `D` dimensions, such as the unknown of a dynamic solve.
pub struct SolidRate<const D: usize>;

/// A scalar field, named by its unit.
pub struct Field<U>(PhantomData<U>);

/// The degrees of freedom of a node in a layout, then those of a node in another layout.
pub struct Coupled<A, B>(PhantomData<(A, B)>);

/// The coordinates of a solid in `D` dimensions, then its temperature, to a node.
pub type SolidThermal<const D: usize> = Coupled<Solid<D>, Field<TemperatureUnit>>;

impl<const D: usize> Layout for Solid<D> {
    const STRIDE: usize = D;
}

impl<const D: usize> Layout for SolidRate<D> {
    const STRIDE: usize = D;
}

impl<U> Layout for Field<U> {
    const STRIDE: usize = 1;
}

impl<A, B> Layout for Coupled<A, B>
where
    A: Layout,
    B: Layout,
{
    const STRIDE: usize = A::STRIDE + B::STRIDE;
}

impl<const D: usize> NodalDof for TensorRank1<D, Current, Length> {
    type Layout = Solid<D>;
}

impl<const D: usize> NodalDof for TensorRank1<D, Current, VelocityUnit> {
    type Layout = SolidRate<D>;
}

impl<U> NodalDof for Quantity<U> {
    type Layout = Field<U>;
}

impl<A, B> NodalDof for TensorTuple<A, B>
where
    A: NodalDof + Tensor,
    B: NodalDof + Tensor,
{
    type Layout = Coupled<A::Layout, B::Layout>;
}

impl<N> NodalState for TensorVector<N>
where
    N: NodalDof,
{
    type Layout = N::Layout;
}

/// A component that a layout has directly.
pub struct Direct;

/// A component that the first layout of a coupled pair has, found by the index in it.
pub struct Left<I>(PhantomData<I>);

/// A component that the second layout of a coupled pair has, found by the index in it.
pub struct Right<I>(PhantomData<I>);

/// A degree of freedom of each node in a layout, at `OFFSET` within its node.
///
/// The index `I` is found by the compiler, from where the component is in a coupled layout.
pub trait Component<L, I>
where
    L: Layout,
{
    const OFFSET: usize;
}

impl<A, B, C, I> Component<Coupled<A, B>, Left<I>> for C
where
    A: Layout,
    B: Layout,
    C: Component<A, I>,
{
    const OFFSET: usize = C::OFFSET;
}

impl<A, B, C, I> Component<Coupled<A, B>, Right<I>> for C
where
    A: Layout,
    B: Layout,
    C: Component<B, I>,
{
    const OFFSET: usize = A::STRIDE + C::OFFSET;
}

/// A component that can be held at a value.
///
/// A displacement is held at a distance from its initial value, and a field at its value.
pub trait Prescribable {
    type Value;
    fn prescription(value: Self::Value) -> Prescription;
}

/// The displacement along an axis.
#[derive(Clone, Copy)]
pub struct Displacement<const AXIS: usize>;

/// The velocity along an axis.
#[derive(Clone, Copy)]
pub struct Velocity<const AXIS: usize>;

/// The temperature.
#[derive(Clone, Copy)]
pub struct Temperature;

/// The electric potential.
#[derive(Clone, Copy)]
pub struct Potential;

/// The molar concentration.
#[derive(Clone, Copy)]
pub struct Concentration;

impl<const D: usize, const AXIS: usize> Component<Solid<D>, Direct> for Displacement<AXIS> {
    const OFFSET: usize = {
        assert!(AXIS < D, "The axis is not one of the dimensions.");
        AXIS
    };
}

impl<const D: usize, const AXIS: usize> Component<SolidRate<D>, Direct> for Velocity<AXIS> {
    const OFFSET: usize = {
        assert!(AXIS < D, "The axis is not one of the dimensions.");
        AXIS
    };
}

impl Component<Field<TemperatureUnit>, Direct> for Temperature {
    const OFFSET: usize = 0;
}

impl Component<Field<ElectricPotential>, Direct> for Potential {
    const OFFSET: usize = 0;
}

impl Component<Field<ConcentrationUnit>, Direct> for Concentration {
    const OFFSET: usize = 0;
}

impl<const AXIS: usize> Prescribable for Displacement<AXIS> {
    type Value = Quantity<Length>;
    fn prescription(value: Self::Value) -> Prescription {
        Prescription::Offset(value.value())
    }
}

impl<const AXIS: usize> Prescribable for Velocity<AXIS> {
    type Value = Quantity<VelocityUnit>;
    fn prescription(value: Self::Value) -> Prescription {
        Prescription::Absolute(value.value())
    }
}

impl Prescribable for Temperature {
    type Value = Quantity<TemperatureUnit>;
    fn prescription(value: Self::Value) -> Prescription {
        Prescription::Absolute(value.value())
    }
}

impl Prescribable for Potential {
    type Value = Quantity<ElectricPotential>;
    fn prescription(value: Self::Value) -> Prescription {
        Prescription::Absolute(value.value())
    }
}

impl Prescribable for Concentration {
    type Value = Quantity<ConcentrationUnit>;
    fn prescription(value: Self::Value) -> Prescription {
        Prescription::Absolute(value.value())
    }
}
