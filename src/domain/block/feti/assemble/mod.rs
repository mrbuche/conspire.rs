#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
        },
    },
    math::{SquareMatrix, Vector},
};
use std::collections::HashMap;

/// Extracts a subdomain's local dense stiffness and force from a real FEM
/// `Block`, filtering to only the elements whose nodes all lie within the
/// subdomain (an element straddling a subdomain boundary doesn't belong to
/// any single subdomain in a non-overlapping decomposition), and scattering
/// their element-local, unit-carrying contributions into subdomain-local,
/// unitless `SquareMatrix`/`Vector` positions.
pub(crate) fn local_stiffness_and_force<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
>(
    block: &Block<C, F, G, M, N, P>,
    nodal_coordinates: &NodalCoordinates<3>,
    subdomain_nodes: &[usize],
) -> Result<(SquareMatrix, Vector), FiniteElementError>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
{
    const D: usize = 3;
    let local_index: HashMap<usize, usize> = subdomain_nodes
        .iter()
        .enumerate()
        .map(|(local, &node)| (node, local))
        .collect();
    let num_local = subdomain_nodes.len() * D;
    let mut stiffness = SquareMatrix::zero(num_local);
    let mut force = Vector::zero(num_local);
    block
        .connectivity()
        .iter()
        .zip(block.elements())
        .filter(|(nodes, _)| nodes.iter().all(|node| local_index.contains_key(node)))
        .try_for_each(|(nodes, element)| {
            let element_coordinates =
                Block::<C, F, G, M, N, P>::element_coordinates(nodal_coordinates, nodes);
            let forces = element.nodal_forces(block.constitutive_model(), &element_coordinates)?;
            let stiffnesses =
                element.nodal_stiffnesses(block.constitutive_model(), &element_coordinates)?;
            nodes.iter().enumerate().for_each(|(a, node_a)| {
                let local_a = local_index[node_a];
                (0..D).for_each(|i| force[D * local_a + i] += forces[a][i].value());
                nodes.iter().enumerate().for_each(|(b, node_b)| {
                    let local_b = local_index[node_b];
                    (0..D).for_each(|i| {
                        (0..D).for_each(|j| {
                            stiffness[D * local_a + i][D * local_b + j] +=
                                stiffnesses[a][b][i][j].value()
                        })
                    })
                })
            });
            Ok::<(), FiniteElementError>(())
        })?;
    Ok((stiffness, force))
}
