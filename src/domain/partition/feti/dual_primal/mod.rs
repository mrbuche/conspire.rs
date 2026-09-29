pub(crate) mod coarse;
pub(crate) mod condense;
pub(crate) mod corner;
pub(crate) mod rigid;
pub(crate) mod rigid_projector;
pub(crate) mod select;
#[cfg(test)]
mod test;

use crate::{geometry::mesh::Partition, math::Scalar};
use std::collections::HashSet;

pub(crate) use corner::{CornerDofs, CornerSelection};

/// Which (global node, component) pairs are prescribed rather than free.
///
/// A fixed DOF is pinned to zero by elimination: it is excluded from both the
/// primal and dual sets, never appears in any local solve or scatter, and so is
/// left at its zero-initialized value.
///
/// A constraint is a linear relation `sum_k a_k u_k = g` between the
/// displacements of some nodes, enforced by a Lagrange multiplier, whose
/// value is the constraint force. A single entry prescribes a displacement,
/// possibly nonzero, and is a row of the interface operator like the
/// continuity of a shared node, unless its node is a corner, where it joins
/// the corner problem. Only as many of these nodes are made corners as
/// remove the rigid-body modes of the subdomains and of the whole block. A
/// constraint of several entries is always in the corner problem, whose
/// nodes are all made corners.
pub struct BoundaryConditions {
    fixed: HashSet<(usize, usize)>,
    constraints: Vec<Constraint>,
}

/// A linear constraint on nodal displacements, `sum_k a_k u_k = value`.
pub(crate) struct Constraint {
    entries: Vec<(usize, usize, Scalar)>,
    value: Scalar,
}

impl BoundaryConditions {
    pub fn new(fixed: Vec<(usize, usize)>) -> Self {
        Self {
            fixed: fixed.into_iter().collect(),
            constraints: Vec::new(),
        }
    }
    pub fn none() -> Self {
        Self::new(Vec::new())
    }
    /// Prescribes a component of a node's displacement by a multiplier.
    pub fn prescribed(self, node: usize, component: usize, value: Scalar) -> Self {
        self.linear(vec![(node, component, 1.0)], value)
    }
    /// Constrains `sum_k a_k u_k = value` over `(node, component, a_k)`
    /// entries by a multiplier.
    pub fn linear(mut self, entries: Vec<(usize, usize, Scalar)>, value: Scalar) -> Self {
        self.constraints.push(Constraint { entries, value });
        self
    }
    pub(crate) fn is_fixed(&self, node: usize, component: usize) -> bool {
        self.fixed.contains(&(node, component))
    }
    pub(crate) fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
    pub(crate) fn fixed(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.fixed.iter().copied()
    }
    /// The constraints with their fixed DOFs dropped, since those are held at
    /// zero already, and those left with nothing to constrain dropped whole.
    pub(crate) fn rows(&self) -> Vec<Row> {
        self.constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                let entries: Vec<_> = constraint
                    .entries
                    .iter()
                    .copied()
                    .filter(|&(node, component, _)| !self.is_fixed(node, component))
                    .collect();
                (!entries.is_empty()).then_some(Row {
                    index,
                    entries,
                    value: constraint.value,
                })
            })
            .collect()
    }
}

/// A constraint on the DOFs that are left free.
pub(crate) struct Row {
    pub(crate) index: usize,
    pub(crate) entries: Vec<(usize, usize, Scalar)>,
    pub(crate) value: Scalar,
}

impl Row {
    /// The one DOF of a row that prescribes a displacement.
    pub(crate) fn single(&self) -> Option<(usize, usize, Scalar)> {
        (self.entries.len() == 1).then(|| self.entries[0])
    }
}

/// A subdomain's local DOFs split into corner (primal) and the rest (dual).
///
/// `primal` numbers corner DOFs in this subdomain's own local numbering;
/// `primal_global` numbers the same DOFs, in the same order, in the global
/// corner-DOF numbering they share with every other subdomain that touches
/// that corner.
pub(crate) struct DualPrimalSplit {
    primal: Vec<usize>,
    primal_global: Vec<usize>,
    dual: Vec<usize>,
}

impl DualPrimalSplit {
    #[cfg(test)]
    pub(crate) fn new(primal: Vec<usize>, dual: Vec<usize>) -> Self {
        let primal_global = primal.clone();
        Self {
            primal,
            primal_global,
            dual,
        }
    }
    fn from_subdomain_nodes(
        subdomain_nodes: &[usize],
        corner_dofs: &CornerDofs,
        boundary_conditions: &BoundaryConditions,
        dimension: usize,
    ) -> Self {
        let mut primal = Vec::new();
        let mut primal_global = Vec::new();
        let mut dual = Vec::new();
        subdomain_nodes
            .iter()
            .enumerate()
            .for_each(|(local, &node)| {
                (0..dimension).for_each(|component| {
                    if boundary_conditions.is_fixed(node, component) {
                        return;
                    }
                    let dof = dimension * local + component;
                    if let Some(global) = corner_dofs.global_index(node, component) {
                        primal.push(dof);
                        primal_global.push(global);
                    } else {
                        dual.push(dof);
                    }
                })
            });
        Self {
            primal,
            primal_global,
            dual,
        }
    }
    pub(crate) fn primal(&self) -> &[usize] {
        &self.primal
    }
    pub(crate) fn primal_global(&self) -> &[usize] {
        &self.primal_global
    }
    pub(crate) fn dual(&self) -> &[usize] {
        &self.dual
    }
}

pub(crate) fn build_splits(
    partition: &Partition,
    corners: &CornerSelection,
    boundary_conditions: &BoundaryConditions,
    dimension: usize,
) -> (Vec<DualPrimalSplit>, CornerDofs) {
    let corner_dofs = CornerDofs::new(corners, boundary_conditions, dimension);
    let splits = partition
        .parts_nodes()
        .iter()
        .map(|nodes| {
            DualPrimalSplit::from_subdomain_nodes(
                nodes,
                &corner_dofs,
                boundary_conditions,
                dimension,
            )
        })
        .collect();
    (splits, corner_dofs)
}
