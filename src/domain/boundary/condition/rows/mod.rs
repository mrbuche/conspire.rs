#[cfg(test)]
mod test;

use super::{Condition, Constrain, ConstrainAt, Enforcement, Static, held};
use crate::{
    domain::boundary::{Component, Layout, Prescribable},
    math::{
        Quantity,
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::Time,
};
use std::marker::PhantomData;

/// Degrees of freedom held at values, each given by its index.
///
/// The number of them is not known when compiling, such as when they are read from rows.
/// The indices are those of the layout, so a degree of freedom that the layout does not have is
/// not found until solving.
pub struct Rows<L> {
    entries: Vec<(usize, Prescription)>,
    enforcement: Enforcement,
    layout: PhantomData<L>,
}

impl<L> Rows<L> {
    pub fn new(enforcement: Enforcement) -> Self {
        Self {
            entries: Vec::new(),
            enforcement,
            layout: PhantomData,
        }
    }
    /// Holds an index at a prescription.
    pub fn hold(mut self, index: usize, prescription: Prescription) -> Self {
        self.entries.push((index, prescription));
        self
    }
    /// Holds a component of a node at a value.
    pub fn add<C, I>(self, _: C, node: usize, value: C::Value) -> Self
    where
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.hold(
            L::STRIDE * node + <C as Component<L, I>>::OFFSET,
            C::prescription(value),
        )
    }
}

impl<L> Condition<L> for Rows<L> {
    fn constraint_at(&self, _: Quantity<Time>) -> EqualityConstraint {
        Static::held_constraint(self)
    }
}

impl<L> Static<L> for Rows<L> {
    fn held_constraint(&self) -> EqualityConstraint {
        held(self.entries.clone(), self.enforcement)
    }
}

impl<L> ConstrainAt<L> for Rows<L> {
    fn constraint_at(&self, _: Quantity<Time>) -> EqualityConstraints {
        vec![Static::held_constraint(self)]
    }
}

impl<L> Constrain<L> for Rows<L> {
    fn constraint(&self) -> EqualityConstraints {
        vec![Static::held_constraint(self)]
    }
}
