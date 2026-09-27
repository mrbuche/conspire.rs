#[cfg(test)]
mod test;

use super::dual_primal::CornerSelection;
use crate::{
    geometry::mesh::Partition,
    math::{Scalar, Vector},
};
use std::collections::HashMap;

pub(crate) struct Interface {
    multipliers: Vec<usize>,
    dofs: Vec<usize>,
    signs: Vec<Scalar>,
}

impl Interface {
    pub(crate) fn multipliers(&self) -> &[usize] {
        &self.multipliers
    }
    pub(crate) fn dofs(&self) -> &[usize] {
        &self.dofs
    }
    pub(crate) fn apply(&self, local: &Vector, num_multipliers: usize) -> Vector {
        let mut global = Vector::zero(num_multipliers);
        self.multipliers
            .iter()
            .zip(self.dofs.iter().zip(self.signs.iter()))
            .for_each(|(&multiplier, (&dof, &sign))| global[multiplier] += sign * local[dof]);
        global
    }
    pub(crate) fn apply_transpose(&self, lambda: &Vector, num_local: usize) -> Vector {
        let mut local = Vector::zero(num_local);
        self.multipliers
            .iter()
            .zip(self.dofs.iter().zip(self.signs.iter()))
            .for_each(|(&multiplier, (&dof, &sign))| local[dof] += sign * lambda[multiplier]);
        local
    }
}

/// Builds a non-redundant jump operator per subdomain: a global node shared by
/// k subdomains contributes k-1 multipliers per dimension, chaining consecutive
/// subdomains (in ascending subdomain index) so continuity is enforced
/// transitively across the whole shared node.
///
/// A corner (primal) node is excluded even where shared: it is already
/// enforced exactly continuous by direct assembly into the coarse problem
/// (see `dual_primal::coarse`), not weakly via a multiplier.
pub(crate) fn build_interfaces(
    partition: &Partition,
    corners: &CornerSelection,
    dimension: usize,
) -> (Vec<Interface>, usize) {
    let num_subdomains = partition.number_of_parts();
    let mut node_occurrences: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    partition
        .parts_nodes()
        .iter()
        .enumerate()
        .for_each(|(subdomain, nodes)| {
            nodes.iter().enumerate().for_each(|(local, &node)| {
                node_occurrences
                    .entry(node)
                    .or_default()
                    .push((subdomain, local));
            })
        });
    let mut multipliers = vec![Vec::new(); num_subdomains];
    let mut dofs = vec![Vec::new(); num_subdomains];
    let mut signs = vec![Vec::new(); num_subdomains];
    let mut num_multipliers = 0;
    let mut shared_nodes: Vec<_> = node_occurrences
        .into_iter()
        .filter(|(node, occurrences)| occurrences.len() > 1 && !corners.contains(*node))
        .collect();
    shared_nodes.sort_unstable_by_key(|&(node, _)| node);
    shared_nodes.into_iter().for_each(|(_, mut occurrences)| {
        occurrences.sort_unstable_by_key(|&(subdomain, _)| subdomain);
        occurrences.windows(2).for_each(|pair| {
            let (subdomain_a, local_a) = pair[0];
            let (subdomain_b, local_b) = pair[1];
            (0..dimension).for_each(|component| {
                multipliers[subdomain_a].push(num_multipliers);
                dofs[subdomain_a].push(dimension * local_a + component);
                signs[subdomain_a].push(1.0);
                multipliers[subdomain_b].push(num_multipliers);
                dofs[subdomain_b].push(dimension * local_b + component);
                signs[subdomain_b].push(-1.0);
                num_multipliers += 1;
            })
        })
    });
    let interfaces = multipliers
        .into_iter()
        .zip(dofs)
        .zip(signs)
        .map(|((multipliers, dofs), signs)| Interface {
            multipliers,
            dofs,
            signs,
        })
        .collect();
    (interfaces, num_multipliers)
}
