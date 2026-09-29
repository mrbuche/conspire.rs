#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh, Patch},
    math::{
        FxHashMap, FxHashSet, Quantity,
        interpolate::{moving_least_squares, quartic_weight},
    },
    units::Length,
};
use std::array::from_fn;

/// Reproducing basis functions on the nodes of a mesh, one for each seed.
///
/// For each seed, the nodes at which its basis function is not zero, with the
/// value there. The function is defined between nodes by linear interpolation
/// over the elements.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    pub seeds: Vec<usize>,
    pub values: Vec<Vec<(usize, f64)>>,
}

fn sorted(mut face: Vec<usize>) -> Vec<usize> {
    face.sort_unstable();
    face
}

/// The radius to which a weight function reaches, which is the distance to the
/// nearest patch boundary that cuts through the mesh, if that is within the radius.
///
/// The parts of the patch boundary that are the mesh boundary are excluded, so
/// weight functions do not shrink away from where the domain simply ends.
fn interior_radius(
    elements: &[(&Connectivity, &[usize])],
    patch: &Patch,
    distances: &[(usize, f64)],
    exterior: &FxHashSet<Vec<usize>>,
    radius: f64,
) -> f64 {
    let mut counts: FxHashMap<Vec<usize>, usize> = FxHashMap::default();
    for &element in &patch.elements {
        let (block, nodes) = elements[element];
        for face in block.element_faces(nodes) {
            *counts.entry(sorted(face)).or_insert(0) += 1;
        }
    }
    counts
        .iter()
        .filter(|(face, count)| **count == 1 && !exterior.contains(*face))
        .flat_map(|(face, _)| face.iter())
        .map(|&node| {
            let at = distances
                .binary_search_by_key(&node, |&(n, _)| n)
                .expect("patch node has a distance");
            distances[at].1
        })
        .fold(radius, f64::min)
}

impl<const D: usize> Mesh<D> {
    /// The reproducing basis of moving least squares on the nodes of the mesh,
    /// with a function for each seed node.
    ///
    /// Each function is supported on the [patch](Mesh::patch) of its seed for the
    /// radius. Its weight is a quartic of the geodesic distance from the seed,
    /// reaching zero at the nearest place the patch is cut off from the rest of
    /// the mesh, and the functions reproduce polynomials up to the degree.
    pub fn reproducing_basis(
        &self,
        seeds: &[usize],
        radius: Quantity<Length>,
        degree: usize,
    ) -> Result<Basis, &'static str> {
        let reach = radius.value_as::<Length>();
        let patches = self.patches(seeds, radius);
        let exterior: FxHashSet<Vec<usize>> =
            self.exterior_faces().into_iter().map(sorted).collect();
        let elements: Vec<(&Connectivity, &[usize])> = self
            .iter()
            .flat_map(|block| block.iter().map(move |element| (block, element)))
            .collect();
        let mut nodes_seeds: FxHashMap<usize, Vec<(usize, f64)>> = FxHashMap::default();
        for (index, (&seed, patch)) in seeds.iter().zip(&patches).enumerate() {
            let distances: Vec<(usize, f64)> = self
                .geodesic_distances_over(seed, &patch.elements)?
                .into_iter()
                .map(|(node, distance)| (node, distance.value_as::<Length>()))
                .collect();
            let cut = interior_radius(&elements, patch, &distances, &exterior, reach);
            if cut <= 0.0 {
                return Err("seed is on the interior boundary of its own patch");
            }
            for &(node, distance) in &distances {
                let weight = quartic_weight(distance / cut);
                if weight > 0.0 {
                    nodes_seeds.entry(node).or_default().push((index, weight));
                }
            }
        }
        let covered = (0..self.number_of_nodes()).all(|node| {
            self.node_element_connectivity()[node].is_empty() || nodes_seeds.contains_key(&node)
        });
        if !covered {
            return Err("seeds do not cover the mesh");
        }
        let point = |node: usize| -> [f64; D] {
            from_fn(|k| self.coordinates()[node][k].value_as::<Length>())
        };
        let mut values = vec![Vec::new(); seeds.len()];
        let mut nodes: Vec<usize> = nodes_seeds.keys().copied().collect();
        nodes.sort_unstable();
        for node in nodes {
            let entries = &nodes_seeds[&node];
            let centers: Vec<[f64; D]> = entries.iter().map(|&(i, _)| point(seeds[i])).collect();
            let weights: Vec<f64> = entries.iter().map(|&(_, w)| w).collect();
            let psi = moving_least_squares(point(node), &centers, &weights, degree)
                .map_err(|_| "too few seeds reach a node; enlarge the radius or add seeds")?;
            entries
                .iter()
                .zip(psi)
                .for_each(|(&(index, _), value)| values[index].push((node, value)));
        }
        Ok(Basis {
            seeds: seeds.to_vec(),
            values,
        })
    }
}
