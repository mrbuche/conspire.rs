#[cfg(test)]
mod test;

use super::dual_primal::CornerSelection;
use crate::{
    geometry::mesh::Partition,
    math::{Scalar, Vector},
};
use std::collections::HashMap;

/// One subdomain's block of the interface (jump) operator.
///
/// A sparse triplet representation: `multipliers[k]` is the shared Lagrange
/// multiplier that local DOF `dofs[k]` participates in, weighted by
/// `signs[k]` (+1 or -1, one sign per side of a shared node, so their
/// difference enforces continuity there). `apply` and `apply_transpose` are
/// `B_s` and `B_s^T` applied without ever being assembled as matrices.
pub(crate) struct Interface {
    multipliers: Vec<usize>,
    dofs: Vec<usize>,
    signs: Vec<Scalar>,
    scaled: Scaled,
}

/// One subdomain's block of the multiplicity-scaled jump operator `B_D`.
///
/// At a node shared by `k` subdomains, `B` chains the copies with `k - 1`
/// multipliers, and `B_D = (B B^T)^-1 B` is the scaling that makes
/// `B_D^T B` take each copy to its deviation from the average over the `k`.
/// Unlike `B`, a multiplier of `B_D` reaches every copy of its node, so
/// this is a set of triplets of its own.
#[derive(Default)]
struct Scaled {
    multipliers: Vec<usize>,
    dofs: Vec<usize>,
    weights: Vec<Scalar>,
}

/// Entry `(multiplier, copy)` of `(B B^T)^-1 B` for `copies` copies of a node
/// chained by `B`, whose `B B^T` is the path Laplacian with a Dirichlet end
/// on each side and so has a closed-form inverse.
fn scaled_weight(multiplier: usize, copy: usize, copies: usize) -> Scalar {
    let inverse = |i: usize, j: usize| {
        let (low, high) = ((i.min(j) + 1) as Scalar, (i.max(j) + 1) as Scalar);
        low * (copies as Scalar - high) / copies as Scalar
    };
    let last = copies - 1;
    let after = if copy < last {
        inverse(multiplier, copy)
    } else {
        0.0
    };
    let before = if copy > 0 {
        inverse(multiplier, copy - 1)
    } else {
        0.0
    };
    after - before
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
    /// `B_D,s` applied to `local`.
    pub(crate) fn apply_scaled(&self, local: &Vector, num_multipliers: usize) -> Vector {
        let mut global = Vector::zero(num_multipliers);
        self.scaled
            .multipliers
            .iter()
            .zip(self.scaled.dofs.iter().zip(self.scaled.weights.iter()))
            .for_each(|(&multiplier, (&dof, &weight))| global[multiplier] += weight * local[dof]);
        global
    }
    /// `B_D,s^T` applied to `lambda`.
    pub(crate) fn apply_transpose_scaled(&self, lambda: &Vector, num_local: usize) -> Vector {
        let mut local = Vector::zero(num_local);
        self.scaled
            .multipliers
            .iter()
            .zip(self.scaled.dofs.iter().zip(self.scaled.weights.iter()))
            .for_each(|(&multiplier, (&dof, &weight))| local[dof] += weight * lambda[multiplier]);
        local
    }
    /// `B_s` applied to `local`, as only the multipliers it is nonzero on.
    pub(crate) fn apply_sparse(&self, local: &Vector) -> Vec<(usize, Scalar)> {
        self.multipliers
            .iter()
            .zip(self.dofs.iter().zip(self.signs.iter()))
            .filter_map(|(&multiplier, (&dof, &sign))| {
                let value = sign * local[dof];
                (value != 0.0).then_some((multiplier, value))
            })
            .collect()
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

/// Builds every subdomain's block of the interface (jump) operator.
///
/// A node shared by k subdomains contributes k-1 multipliers per dimension,
/// chaining consecutive subdomains (by ascending index) so continuity is
/// enforced transitively across the whole shared node. A corner (primal)
/// node is excluded even where shared: it's already enforced exactly
/// continuous by direct assembly into the coarse problem, not weakly
/// through a multiplier.
pub(crate) fn build_interfaces(
    partition: &Partition,
    corners: &CornerSelection,
    dimension: usize,
) -> (Vec<Interface>, usize) {
    let num_subdomains = partition.number_of_parts();
    let mut node_occurrences = HashMap::<usize, Vec<(usize, usize)>>::new();
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
    let mut scaled: Vec<Scaled> = (0..num_subdomains).map(|_| Scaled::default()).collect();
    let mut num_multipliers = 0;
    let mut shared_nodes: Vec<_> = node_occurrences
        .into_iter()
        .filter(|(node, occurrences)| occurrences.len() > 1 && !corners.contains(*node))
        .collect();
    shared_nodes.sort_unstable_by_key(|&(node, _)| node);
    shared_nodes.into_iter().for_each(|(_, mut occurrences)| {
        occurrences.sort_unstable_by_key(|&(subdomain, _)| subdomain);
        let first = num_multipliers;
        let copies = occurrences.len();
        (0..copies - 1).for_each(|multiplier| {
            (0..dimension).for_each(|component| {
                occurrences
                    .iter()
                    .enumerate()
                    .for_each(|(copy, &(subdomain, local))| {
                        let weight = scaled_weight(multiplier, copy, copies);
                        if weight != 0.0 {
                            scaled[subdomain]
                                .multipliers
                                .push(first + dimension * multiplier + component);
                            scaled[subdomain].dofs.push(dimension * local + component);
                            scaled[subdomain].weights.push(weight);
                        }
                    })
            })
        });
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
        .zip(scaled)
        .map(|(((multipliers, dofs), signs), scaled)| Interface {
            multipliers,
            dofs,
            signs,
            scaled,
        })
        .collect();
    (interfaces, num_multipliers)
}
