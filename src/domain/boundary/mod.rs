//! Boundary conditions on the degrees of freedom of a model.

mod condition;

pub use condition::BoundaryConditions;

use crate::units::{Charge, Current, MolarFlowRate, Power};
use std::marker::PhantomData;

/// How the degrees of freedom are laid out, `STRIDE` to a node, indexed as `STRIDE * node + offset`.
pub trait Layout {
    const STRIDE: usize;
}

/// A degree of freedom of each node in a layout, at `OFFSET` within its node.
pub trait Component<L>
where
    L: Layout,
{
    const OFFSET: usize;
}

/// A scalar field, named by the unit of the residual of its balance law.
pub struct Field<L>(PhantomData<L>);

/// The coordinates of a solid in `D` dimensions.
pub struct Solid<const D: usize>;

/// The coordinates of a solid in `D` dimensions, then its temperature, to a node.
pub struct SolidThermal<const D: usize>;

impl<L> Layout for Field<L> {
    const STRIDE: usize = 1;
}

impl<const D: usize> Layout for Solid<D> {
    const STRIDE: usize = D;
}

impl<const D: usize> Layout for SolidThermal<D> {
    const STRIDE: usize = D + 1;
}

/// The temperature.
pub struct Temperature;

/// The electric potential.
pub struct Potential;

/// The molar concentration.
pub struct Concentration;

/// The displacement along an axis.
pub struct Displacement<const AXIS: usize>;

impl<const D: usize, const AXIS: usize> Component<Solid<D>> for Displacement<AXIS> {
    const OFFSET: usize = {
        assert!(AXIS < D, "The axis is not one of the dimensions.");
        AXIS
    };
}

impl<const D: usize, const AXIS: usize> Component<SolidThermal<D>> for Displacement<AXIS> {
    const OFFSET: usize = {
        assert!(AXIS < D, "The axis is not one of the dimensions.");
        AXIS
    };
}

impl<const D: usize> Component<SolidThermal<D>> for Temperature {
    const OFFSET: usize = D;
}

impl Component<Field<Power>> for Temperature {
    const OFFSET: usize = 0;
}

impl Component<Field<Charge>> for Potential {
    const OFFSET: usize = 0;
}

impl Component<Field<Current>> for Potential {
    const OFFSET: usize = 0;
}

impl Component<Field<MolarFlowRate>> for Concentration {
    const OFFSET: usize = 0;
}
