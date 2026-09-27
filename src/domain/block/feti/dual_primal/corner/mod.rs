#[cfg(test)]
mod test;

use super::BoundaryConditions;
use crate::geometry::mesh::Partition;
use std::collections::HashMap;

/// Which nodes are corners.
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
    pub(crate) fn contains(&self, node: usize) -> bool {
        self.nodes.binary_search(&node).is_ok()
    }
    pub(crate) fn nodes(&self) -> &[usize] {
        &self.nodes
    }
}

/// Which corner DOFs survive as free primal unknowns.
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
