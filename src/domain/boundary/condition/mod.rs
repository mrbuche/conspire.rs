#[cfg(test)]
mod test;

use crate::{
    domain::boundary::{Component, Layout},
    geometry::mesh::NodeSet,
    math::optimize::EqualityConstraint,
};
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
enum Target<'m> {
    Set(&'m NodeSet),
    Nodes(Vec<usize>),
}

/// A component of some nodes held fixed, as `stride * node + offset`.
///
/// The nodes are a set that the condition borrows, or are listed.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryCondition<'m> {
    target: Target<'m>,
    stride: usize,
    offset: usize,
}

impl BoundaryCondition<'_> {
    /// The degrees of freedom held.
    pub fn indices(&self) -> Vec<usize> {
        let nodes = match &self.target {
            Target::Set(set) => set.nodes(),
            Target::Nodes(nodes) => nodes,
        };
        nodes
            .iter()
            .map(|&node| self.stride * node + self.offset)
            .collect()
    }
}

/// Degrees of freedom held fixed, chosen by node and component.
///
/// The components a layout has are the only ones that can be fixed, and an axis that a layout
/// does not have is an error when compiling. The conditions borrow the sets they fix, such as
/// those of the model they are for, so they cannot outlive them.
pub struct BoundaryConditions<'m, L> {
    conditions: Vec<BoundaryCondition<'m>>,
    layout: PhantomData<L>,
}

impl<L> Default for BoundaryConditions<'_, L> {
    fn default() -> Self {
        Self {
            conditions: Vec::new(),
            layout: PhantomData,
        }
    }
}

impl<'m, L> BoundaryConditions<'m, L>
where
    L: Layout,
{
    pub fn new() -> Self {
        Self::default()
    }
    fn with<C>(mut self, target: Target<'m>) -> Self
    where
        C: Component<L>,
    {
        self.conditions.push(BoundaryCondition {
            target,
            stride: L::STRIDE,
            offset: C::OFFSET,
        });
        self
    }
    /// Holds a component of every node in a set at its initial value.
    pub fn fix<C>(self, nodes: &'m NodeSet) -> Self
    where
        C: Component<L>,
    {
        self.with::<C>(Target::Set(nodes))
    }
    /// Holds a component of the listed nodes at its initial value.
    pub fn fix_nodes<C>(self, nodes: impl IntoIterator<Item = usize>) -> Self
    where
        C: Component<L>,
    {
        self.with::<C>(Target::Nodes(nodes.into_iter().collect()))
    }
    /// Holds a component of a node at its initial value.
    pub fn fix_node<C>(self, node: usize) -> Self
    where
        C: Component<L>,
    {
        self.with::<C>(Target::Nodes(vec![node]))
    }
    /// The conditions, in the order they were made.
    pub fn conditions(&self) -> &[BoundaryCondition<'m>] {
        &self.conditions
    }
    /// The constraint holding the degrees of freedom, in ascending order and without repeats.
    pub fn constraint(&self) -> EqualityConstraint {
        let mut fixed: Vec<usize> = self
            .conditions
            .iter()
            .flat_map(BoundaryCondition::indices)
            .collect();
        fixed.sort_unstable();
        fixed.dedup();
        EqualityConstraint::Fixed(fixed)
    }
}
