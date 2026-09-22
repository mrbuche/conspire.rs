pub(crate) mod coarse;
pub(crate) mod condense;
#[cfg(test)]
mod test;

use super::interface::Partition;
use std::collections::{HashMap, HashSet};

/// Which (global node, component) pairs are prescribed rather than free.
/// Only zero-displacement (homogeneous) Dirichlet conditions are supported:
/// a fixed DOF is simply excluded from both the primal and dual sets, never
/// appears in any local solve or scatter, and so is left at its
/// zero-initialized value everywhere — which is exactly correct for a
/// zero prescribed displacement, and needs no force-correction term the way
/// a nonzero prescribed value would.
pub(crate) struct BoundaryConditions {
    fixed: HashSet<(usize, usize)>,
}

impl BoundaryConditions {
    pub(crate) fn new(fixed: Vec<(usize, usize)>) -> Self {
        Self {
            fixed: fixed.into_iter().collect(),
        }
    }
    pub(crate) fn none() -> Self {
        Self {
            fixed: HashSet::new(),
        }
    }
    fn is_fixed(&self, node: usize, component: usize) -> bool {
        self.fixed.contains(&(node, component))
    }
}

pub(crate) struct CornerSelection {
    nodes: Vec<usize>,
}

impl CornerSelection {
    pub(crate) fn new(mut nodes: Vec<usize>) -> Self {
        nodes.sort_unstable();
        nodes.dedup();
        Self { nodes }
    }
    /// Standard FETI-DP heuristic: a node shared by three or more subdomains
    /// (a cross point) is pinned primal; a node shared by exactly two stays
    /// on the dual (Lagrange-multiplier) interface.
    pub(crate) fn from_partition(partition: &Partition) -> Self {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        partition.subdomains_nodes().iter().for_each(|nodes| {
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

/// The global numbering of corner DOFs that actually survive as free primal
/// unknowns, combining `CornerSelection` (which NODES are corners) with
/// `BoundaryConditions` (which of their components are pinned rather than
/// free). A `(node, component)` pair only gets a slot here if the node is a
/// corner AND that component isn't fixed — unlike numbering every corner
/// node's `dimension` components unconditionally, this keeps the assembled
/// coarse problem exactly the size of its genuinely free DOFs, so a
/// boundary condition on a corner component can never leave a
/// permanently-zero (singular) row/column behind.
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
    fn global_index(&self, node: usize, component: usize) -> Option<usize> {
        self.index.get(&(node, component)).copied()
    }
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

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
        .subdomains_nodes()
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
