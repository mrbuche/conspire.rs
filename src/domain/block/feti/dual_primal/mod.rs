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
    fn global_index(&self, node: usize) -> Option<usize> {
        self.nodes.binary_search(&node).ok()
    }
    pub(crate) fn nodes(&self) -> &[usize] {
        &self.nodes
    }
    pub(crate) fn num_corners(&self) -> usize {
        self.nodes.len()
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
        corners: &CornerSelection,
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
                    if let Some(global_node) = corners.global_index(node) {
                        primal.push(dof);
                        primal_global.push(dimension * global_node + component);
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
) -> Vec<DualPrimalSplit> {
    partition
        .subdomains_nodes()
        .iter()
        .map(|nodes| {
            DualPrimalSplit::from_subdomain_nodes(nodes, corners, boundary_conditions, dimension)
        })
        .collect()
}
