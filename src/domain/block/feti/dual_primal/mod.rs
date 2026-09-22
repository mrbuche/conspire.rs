pub(crate) mod coarse;
pub(crate) mod condense;
#[cfg(test)]
mod test;

use super::{Subdomain, interface::Partition};
use crate::math::Vector;
use std::collections::HashMap;

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
    fn contains(&self, node: usize) -> bool {
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
        dimension: usize,
    ) -> Self {
        let mut primal = Vec::new();
        let mut primal_global = Vec::new();
        let mut dual = Vec::new();
        subdomain_nodes
            .iter()
            .enumerate()
            .for_each(|(local, &node)| {
                let dofs = (dimension * local)..(dimension * (local + 1));
                if let Some(global_node) = corners.global_index(node) {
                    dofs.for_each(|dof| {
                        let component = dof - dimension * local;
                        primal.push(dof);
                        primal_global.push(dimension * global_node + component);
                    });
                } else {
                    dual.extend(dofs);
                }
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
    dimension: usize,
) -> Vec<DualPrimalSplit> {
    partition
        .subdomains_nodes()
        .iter()
        .map(|nodes| DualPrimalSplit::from_subdomain_nodes(nodes, corners, dimension))
        .collect()
}

pub(crate) fn solve<B>(_subdomains: &[Subdomain<B>], _splits: &[DualPrimalSplit]) -> Vector {
    todo!(
        "assemble the corner coarse problem, condense to non-singular local K_s, projected PCG on the dual DOFs"
    )
}
