#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{
        Connectivity, Mesh, Patch, differential::geodesic::geodesic_distances_among,
        simplex::Simplex,
    },
    math::{
        FxHashMap, FxHashSet, Quantity,
        interpolate::{moving_least_squares, quartic_weight},
    },
    units::Length,
};
use std::array::from_fn;

const NOT_SIMPLICIAL: &str = "reproducing bases require a triangular or tetrahedral mesh";

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

/// The faces of every element, numbered once so that a patch can count how many
/// of its elements touch each face without hashing the faces again.
struct Faces {
    elements: Vec<usize>,
    members: Vec<usize>,
    nodes: Vec<Vec<usize>>,
    exterior: Vec<bool>,
}

impl Faces {
    fn new(elements: &[(&Connectivity, &[usize])], exterior: &FxHashSet<Vec<usize>>) -> Self {
        let mut numbers = FxHashMap::<Vec<usize>, usize>::default();
        let mut nodes = Vec::new();
        let mut exterior_faces = Vec::new();
        let mut pointers = vec![0];
        let mut members = Vec::new();
        for &(block, element) in elements {
            for face in block.element_faces(element) {
                let face = sorted(face);
                let number = *numbers.entry(face).or_insert_with_key(|face| {
                    nodes.push(face.clone());
                    exterior_faces.push(exterior.contains(face));
                    nodes.len() - 1
                });
                members.push(number);
            }
            pointers.push(members.len());
        }
        Self {
            elements: pointers,
            members,
            nodes,
            exterior: exterior_faces,
        }
    }
    fn of(&self, element: usize) -> &[usize] {
        &self.members[self.elements[element]..self.elements[element + 1]]
    }
}

/// The radius to which a weight function reaches, which is the distance to the
/// nearest patch boundary that cuts through the mesh, if that is within the radius.
///
/// The parts of the patch boundary that are the mesh boundary are excluded, so
/// weight functions do not shrink away from where the domain simply ends.
fn interior_radius(
    faces: &Faces,
    counts: &mut [u8],
    patch: &Patch,
    distances: &[(usize, f64)],
    radius: f64,
) -> f64 {
    for &element in &patch.elements {
        for &face in faces.of(element) {
            counts[face] += 1;
        }
    }
    let mut cut = radius;
    for &element in &patch.elements {
        for &face in faces.of(element) {
            if counts[face] == 1 && !faces.exterior[face] {
                for &node in &faces.nodes[face] {
                    let at = distances
                        .binary_search_by_key(&node, |&(n, _)| n)
                        .expect("patch node has a distance");
                    cut = cut.min(distances[at].1);
                }
            }
        }
    }
    for &element in &patch.elements {
        for &face in faces.of(element) {
            counts[face] = 0;
        }
    }
    cut
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
        let elements: Vec<usize> = (0..self.number_of_elements()).collect();
        if let Some(triangles) = self.simplices_over::<3>(&elements) {
            basis(self, &triangles, seeds, radius, degree)
        } else if let Some(tetrahedra) = self.simplices_over::<4>(&elements) {
            basis(self, &tetrahedra, seeds, radius, degree)
        } else {
            Err(NOT_SIMPLICIAL)
        }
    }
}

fn basis<const D: usize, const N: usize>(
    mesh: &Mesh<D>,
    simplices: &[Simplex<D, N>],
    seeds: &[usize],
    radius: Quantity<Length>,
    degree: usize,
) -> Result<Basis, &'static str> {
    let reach = radius.value_as::<Length>();
    let patches = mesh.patches(seeds, radius);
    let exterior: FxHashSet<Vec<usize>> = mesh.exterior_faces().into_iter().map(sorted).collect();
    let elements: Vec<(&Connectivity, &[usize])> = mesh
        .iter()
        .flat_map(|block| block.iter().map(move |element| (block, element)))
        .collect();
    let faces = Faces::new(&elements, &exterior);
    let mut counts = vec![0_u8; faces.nodes.len()];
    let mut nodes_seeds = FxHashMap::<usize, Vec<(usize, f64)>>::default();
    for (index, (&seed, patch)) in seeds.iter().zip(&patches).enumerate() {
        let distances: Vec<(usize, f64)> =
            geodesic_distances_among(mesh, seed, simplices, &patch.elements)?
                .into_iter()
                .map(|(node, distance)| (node, distance.value_as::<Length>()))
                .collect();
        let cut = interior_radius(&faces, &mut counts, patch, &distances, reach);
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
    let covered = (0..mesh.number_of_nodes()).all(|node| {
        mesh.node_element_connectivity()[node].is_empty() || nodes_seeds.contains_key(&node)
    });
    if !covered {
        return Err("seeds do not cover the mesh");
    }
    let point =
        |node: usize| -> [f64; D] { from_fn(|k| mesh.coordinates()[node][k].value_as::<Length>()) };
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
