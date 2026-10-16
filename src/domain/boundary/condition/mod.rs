#[cfg(test)]
mod test;

use crate::{
    domain::boundary::{Component, Layout, Prescribable},
    geometry::mesh::NodeSet,
    math::optimize::{EqualityConstraint, Prescription},
};
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq)]
enum Target<'m> {
    Set(&'m NodeSet),
    Nodes(Vec<usize>),
}

/// A component of some nodes held fixed or at a value, as `stride * node + offset`.
///
/// The nodes are a set that the condition borrows, or are listed.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryCondition<'m> {
    target: Target<'m>,
    stride: usize,
    offset: usize,
    prescription: Prescription,
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
    /// What the degrees of freedom are held at.
    pub fn prescription(&self) -> Prescription {
        self.prescription
    }
}

/// Degrees of freedom held fixed or at a value, chosen by node and component.
///
/// The components a layout has are the only ones that can be held, and an axis that a layout
/// does not have is an error when compiling. The conditions borrow the sets they hold, such as
/// those of the model they are for, so they cannot outlive them. How what is held is enforced
/// is up to the solver.
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
    fn with<C, I>(mut self, target: Target<'m>, prescription: Prescription) -> Self
    where
        C: Component<L, I>,
    {
        self.conditions.push(BoundaryCondition {
            target,
            stride: L::STRIDE,
            offset: <C as Component<L, I>>::OFFSET,
            prescription,
        });
        self
    }
    /// Holds a component of every node in a set at its initial value.
    pub fn fix<C, I>(self, _: C, nodes: &'m NodeSet) -> Self
    where
        C: Component<L, I>,
    {
        self.with::<C, I>(Target::Set(nodes), Prescription::Initial)
    }
    /// Holds a component of the listed nodes at its initial value.
    pub fn fix_nodes<C, I>(self, _: C, nodes: impl IntoIterator<Item = usize>) -> Self
    where
        C: Component<L, I>,
    {
        self.with::<C, I>(
            Target::Nodes(nodes.into_iter().collect()),
            Prescription::Initial,
        )
    }
    /// Holds a component of a node at its initial value.
    pub fn fix_node<C, I>(self, _: C, node: usize) -> Self
    where
        C: Component<L, I>,
    {
        self.with::<C, I>(Target::Nodes(vec![node]), Prescription::Initial)
    }
    /// Holds a component of every node in a set at a value.
    pub fn prescribe<C, I>(self, _: C, nodes: &'m NodeSet, value: C::Value) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.with::<C, I>(Target::Set(nodes), C::prescription(value))
    }
    /// Holds a component of the listed nodes at a value.
    pub fn prescribe_nodes<C, I>(
        self,
        _: C,
        nodes: impl IntoIterator<Item = usize>,
        value: C::Value,
    ) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.with::<C, I>(
            Target::Nodes(nodes.into_iter().collect()),
            C::prescription(value),
        )
    }
    /// Holds a component of a node at a value.
    pub fn prescribe_node<C, I>(self, _: C, node: usize, value: C::Value) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.with::<C, I>(Target::Nodes(vec![node]), C::prescription(value))
    }
    /// The conditions, in the order they were made.
    pub fn conditions(&self) -> &[BoundaryCondition<'m>] {
        &self.conditions
    }
    /// The constraint holding the degrees of freedom.
    ///
    /// It is fixed, with the indices in ascending order and without repeats, if every condition
    /// holds its degrees of freedom at their initial values, and prescribed otherwise.
    pub fn constraint(&self) -> EqualityConstraint {
        if self
            .conditions
            .iter()
            .all(|condition| condition.prescription == Prescription::Initial)
        {
            let mut fixed: Vec<usize> = self
                .conditions
                .iter()
                .flat_map(BoundaryCondition::indices)
                .collect();
            fixed.sort_unstable();
            fixed.dedup();
            EqualityConstraint::Fixed(fixed)
        } else {
            EqualityConstraint::Prescribed(
                self.conditions
                    .iter()
                    .flat_map(|condition| {
                        let prescription = condition.prescription;
                        condition
                            .indices()
                            .into_iter()
                            .map(move |index| (index, prescription))
                    })
                    .collect(),
            )
        }
    }
}
