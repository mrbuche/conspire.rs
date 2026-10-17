#[cfg(test)]
mod test;

use crate::{
    domain::boundary::{Component, Layout, Prescribable},
    geometry::mesh::NodeSet,
    math::optimize::{EqualityConstraint, EqualityConstraints, Prescription},
};
use std::marker::PhantomData;

/// How degrees of freedom are held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Enforcement {
    /// They are eliminated from the system.
    Elimination,
    /// They are held by Lagrange multipliers.
    Multipliers,
}

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
    enforcement: Enforcement,
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
    /// How the degrees of freedom are held.
    pub fn enforcement(&self) -> Enforcement {
        self.enforcement
    }
}

/// Degrees of freedom held fixed or at a value, chosen by node and component.
///
/// The components a layout has are the only ones that can be held, and an axis that a layout
/// does not have is an error when compiling. The conditions borrow the sets they hold, such as
/// those of the model they are for, so they cannot outlive them. Each is eliminated from the
/// system unless it is made to be held by multipliers.
///
/// Constraints that are not on a component of some nodes, such as one tying degrees of
/// freedom together, are added as they are, and are not checked against the layout.
pub struct BoundaryConditions<'m, L> {
    conditions: Vec<BoundaryCondition<'m>>,
    constraints: EqualityConstraints,
    layout: PhantomData<L>,
}

impl<L> Default for BoundaryConditions<'_, L> {
    fn default() -> Self {
        Self {
            conditions: Vec::new(),
            constraints: Vec::new(),
            layout: PhantomData,
        }
    }
}

impl<L> From<EqualityConstraints> for BoundaryConditions<'_, L> {
    fn from(constraints: EqualityConstraints) -> Self {
        Self {
            conditions: Vec::new(),
            constraints,
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
    fn hold<C, I>(
        mut self,
        target: Target<'m>,
        prescription: Prescription,
        enforcement: Enforcement,
    ) -> Self
    where
        C: Component<L, I>,
    {
        self.conditions.push(BoundaryCondition {
            target,
            stride: L::STRIDE,
            offset: <C as Component<L, I>>::OFFSET,
            prescription,
            enforcement,
        });
        self
    }
    /// Holds a component of every node in a set at its initial value.
    pub fn fix<C, I>(self, component: C, nodes: &'m NodeSet) -> Self
    where
        C: Component<L, I>,
    {
        self.fix_by(Enforcement::Elimination, component, nodes)
    }
    /// Holds a component of every node in a set at its initial value, the way given.
    pub fn fix_by<C, I>(self, enforcement: Enforcement, _: C, nodes: &'m NodeSet) -> Self
    where
        C: Component<L, I>,
    {
        self.hold::<C, I>(Target::Set(nodes), Prescription::Initial, enforcement)
    }
    /// Holds a component of the listed nodes at its initial value.
    pub fn fix_nodes<C, I>(self, component: C, nodes: impl IntoIterator<Item = usize>) -> Self
    where
        C: Component<L, I>,
    {
        self.fix_nodes_by(Enforcement::Elimination, component, nodes)
    }
    /// Holds a component of the listed nodes at its initial value, the way given.
    pub fn fix_nodes_by<C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: impl IntoIterator<Item = usize>,
    ) -> Self
    where
        C: Component<L, I>,
    {
        self.hold::<C, I>(
            Target::Nodes(nodes.into_iter().collect()),
            Prescription::Initial,
            enforcement,
        )
    }
    /// Holds a component of a node at its initial value.
    pub fn fix_node<C, I>(self, component: C, node: usize) -> Self
    where
        C: Component<L, I>,
    {
        self.fix_node_by(Enforcement::Elimination, component, node)
    }
    /// Holds a component of a node at its initial value, the way given.
    pub fn fix_node_by<C, I>(self, enforcement: Enforcement, _: C, node: usize) -> Self
    where
        C: Component<L, I>,
    {
        self.hold::<C, I>(
            Target::Nodes(vec![node]),
            Prescription::Initial,
            enforcement,
        )
    }
    /// Holds a component of every node in a set at a value.
    pub fn prescribe<C, I>(self, component: C, nodes: &'m NodeSet, value: C::Value) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of every node in a set at a value, the way given.
    pub fn prescribe_by<C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: &'m NodeSet,
        value: C::Value,
    ) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.hold::<C, I>(Target::Set(nodes), C::prescription(value), enforcement)
    }
    /// Holds a component of the listed nodes at a value.
    pub fn prescribe_nodes<C, I>(
        self,
        component: C,
        nodes: impl IntoIterator<Item = usize>,
        value: C::Value,
    ) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_nodes_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of the listed nodes at a value, the way given.
    pub fn prescribe_nodes_by<C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: impl IntoIterator<Item = usize>,
        value: C::Value,
    ) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.hold::<C, I>(
            Target::Nodes(nodes.into_iter().collect()),
            C::prescription(value),
            enforcement,
        )
    }
    /// Holds a component of a node at a value.
    pub fn prescribe_node<C, I>(self, component: C, node: usize, value: C::Value) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_node_by(Enforcement::Elimination, component, node, value)
    }
    /// Holds a component of a node at a value, the way given.
    pub fn prescribe_node_by<C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        node: usize,
        value: C::Value,
    ) -> Self
    where
        C: Component<L, I> + Prescribable,
    {
        self.hold::<C, I>(
            Target::Nodes(vec![node]),
            C::prescription(value),
            enforcement,
        )
    }
    /// Adds a constraint as it is, on the degrees of freedom as `stride * node + offset`.
    pub fn with_constraint(mut self, constraint: EqualityConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
    /// The conditions, in the order they were made.
    pub fn conditions(&self) -> &[BoundaryCondition<'m>] {
        &self.conditions
    }
    /// The constraints holding the degrees of freedom, one for each condition and then those
    /// added as they are.
    ///
    /// A condition eliminated at its initial values is fixed, with the indices in ascending
    /// order and without repeats, and any other eliminated is prescribed. One held by
    /// multipliers has the indices as they are.
    pub fn constraint(&self) -> EqualityConstraints {
        self.conditions
            .iter()
            .map(|condition| {
                let mut indices = condition.indices();
                let prescription = condition.prescription;
                match condition.enforcement {
                    Enforcement::Multipliers => EqualityConstraint::Multipliers(
                        indices
                            .into_iter()
                            .map(|index| (index, prescription))
                            .collect(),
                    ),
                    Enforcement::Elimination if prescription == Prescription::Initial => {
                        indices.sort_unstable();
                        indices.dedup();
                        EqualityConstraint::Fixed(indices)
                    }
                    Enforcement::Elimination => EqualityConstraint::Prescribed(
                        indices
                            .into_iter()
                            .map(|index| (index, prescription))
                            .collect(),
                    ),
                }
            })
            .chain(self.constraints.iter().cloned())
            .collect()
    }
}
