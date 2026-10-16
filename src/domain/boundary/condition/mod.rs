#[cfg(test)]
mod test;

use crate::{
    domain::boundary::{Component, Layout},
    geometry::mesh::NodeSet,
    math::optimize::EqualityConstraint,
};
use std::marker::PhantomData;

/// Degrees of freedom held fixed, chosen by node and component.
///
/// The components a layout has are the only ones that can be fixed, and an axis that a layout
/// does not have is an error when compiling.
pub struct BoundaryConditions<L> {
    fixed: Vec<usize>,
    layout: PhantomData<L>,
}

impl<L> Default for BoundaryConditions<L> {
    fn default() -> Self {
        Self {
            fixed: Vec::new(),
            layout: PhantomData,
        }
    }
}

impl<L> BoundaryConditions<L>
where
    L: Layout,
{
    pub fn new() -> Self {
        Self::default()
    }
    /// Holds a component of every node in a set at its initial value.
    pub fn fix<C>(mut self, nodes: &NodeSet) -> Self
    where
        C: Component<L>,
    {
        self.fixed.extend(
            nodes
                .nodes()
                .iter()
                .map(|&node| L::STRIDE * node + C::OFFSET),
        );
        self
    }
    /// Holds a component of a node at its initial value.
    pub fn fix_node<C>(mut self, node: usize) -> Self
    where
        C: Component<L>,
    {
        self.fixed.push(L::STRIDE * node + C::OFFSET);
        self
    }
    /// The constraint holding the degrees of freedom, in ascending order and without repeats.
    pub fn constraint(&self) -> EqualityConstraint {
        let mut fixed = self.fixed.clone();
        fixed.sort_unstable();
        fixed.dedup();
        EqualityConstraint::Fixed(fixed)
    }
}
