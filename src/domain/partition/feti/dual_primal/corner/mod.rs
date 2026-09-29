#[cfg(test)]
mod test;

use super::BoundaryConditions;
use crate::geometry::mesh::Partition;
use std::collections::HashMap;

/// Which nodes are corners.
///
/// A corner node is enforced exactly continuous by direct assembly into the
/// coarse problem, rather than weakly through a Lagrange multiplier like the
/// rest of the interface. The standard FETI-DP heuristic makes a node
/// primal (a corner) when it's shared by three or more subdomains, since
/// that's what a subdomain needs, alongside boundary conditions, to remove
/// every one of its own rigid-body modes.
pub(crate) struct CornerSelection {
    nodes: Vec<usize>,
}

impl CornerSelection {
    pub(crate) fn new(mut nodes: Vec<usize>) -> Self {
        nodes.sort_unstable();
        nodes.dedup();
        Self { nodes }
    }
    pub(crate) fn from_partition(partition: &Partition) -> Self {
        let mut counts = HashMap::<usize, usize>::new();
        partition.parts_nodes().iter().for_each(|nodes| {
            nodes.iter().for_each(|&node| {
                *counts.entry(node).or_insert(0) += 1;
            })
        });
        Self::new(
            counts
                .into_iter()
                .filter(|&(_, count)| count >= 3)
                .map(|(node, _)| node)
                .collect(),
        )
    }
    pub(crate) fn with_nodes(self, nodes: impl Iterator<Item = usize>) -> Self {
        Self::new(self.nodes.into_iter().chain(nodes).collect())
    }
    pub(crate) fn contains(&self, node: usize) -> bool {
        self.nodes.binary_search(&node).is_ok()
    }
    pub(crate) fn nodes(&self) -> &[usize] {
        &self.nodes
    }
}

/// Which corner DOFs survive as free primal unknowns.
///
/// Combines `CornerSelection` (which nodes are corners) with
/// `BoundaryConditions` (which of their components are pinned rather than
/// free): a `(node, component)` pair only gets a slot here if the node is a
/// corner AND that component isn't fixed. Numbering every corner node's
/// components unconditionally would instead leave a permanently-zero
/// (singular) row/column in the assembled coarse problem wherever a
/// boundary condition pins a corner component.
pub(crate) struct CornerDofs {
    index: HashMap<(usize, usize), usize>,
    count: usize,
}

impl CornerDofs {
    pub(crate) fn new(
        corners: &CornerSelection,
        boundary_conditions: &BoundaryConditions,
        dimension: usize,
    ) -> Self {
        let mut index = HashMap::new();
        let mut count = 0;
        corners.nodes().iter().for_each(|&node| {
            (0..dimension).for_each(|component| {
                if !boundary_conditions.is_fixed(node, component) {
                    index.insert((node, component), count);
                    count += 1;
                }
            })
        });
        Self { index, count }
    }
    pub(crate) fn global_index(&self, node: usize, component: usize) -> Option<usize> {
        self.index.get(&(node, component)).copied()
    }
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}
