#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{
        Mesh,
        simplex::{Simplex, dot},
    },
    math::{FxHashMap, Quantity, Tensor, Vector, sparse::SparseSolver},
    units::Length,
};
use std::array::from_fn;

type Entries = FxHashMap<(usize, usize), f64>;

const NOT_SIMPLICIAL: &str = "geodesic distances require a triangular or tetrahedral mesh";

fn add(entries: &mut Entries, i: usize, j: usize, value: f64) {
    *entries.entry((i, j)).or_insert(0.0) += value;
}

fn solve(
    solver: &SparseSolver,
    value: impl Fn(usize, usize) -> f64,
    b: &Vector,
) -> Result<Vector, &'static str> {
    solver
        .solve(value, b)
        .map_err(|_| "geodesic linear solve failed")
}

fn heat<const D: usize, const N: usize>(
    mesh: &Mesh<D>,
    source: usize,
    simplices: &[Simplex<D, N>],
) -> Result<Vec<(usize, Quantity<Length>)>, &'static str> {
    let coordinates = mesh.coordinates();
    let point =
        |node: usize| -> [f64; D] { from_fn(|k| coordinates[node][k].value_as::<Length>()) };
    let mut nodes: Vec<usize> = simplices.iter().flat_map(|s| s.nodes).collect();
    nodes.sort_unstable();
    nodes.dedup();
    let source = nodes
        .binary_search(&source)
        .map_err(|_| "source node is not in the elements")?;
    let local = |node: usize| nodes.binary_search(&node).expect("node in elements");
    let n = nodes.len();
    let mut stiffness = Entries::default();
    let mut mass = vec![0.0; n];
    let (mut length, mut count) = (0.0, 0);
    for simplex in simplices {
        let ids = simplex.nodes.map(local);
        for a in 0..N {
            mass[ids[a]] += simplex.volume / N as f64;
            for b in a + 1..N {
                let weight = dot(&simplex.gradients[a], &simplex.gradients[b]) * simplex.volume;
                add(&mut stiffness, ids[a], ids[b], weight);
                add(&mut stiffness, ids[b], ids[a], weight);
                add(&mut stiffness, ids[a], ids[a], -weight);
                add(&mut stiffness, ids[b], ids[b], -weight);
                let (p, q) = (point(simplex.nodes[a]), point(simplex.nodes[b]));
                length += (0..D).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt();
                count += 1;
            }
        }
    }
    let time = (length / count as f64).powi(2);
    let mut heat: Entries = stiffness
        .iter()
        .map(|(&key, &value)| (key, time * value))
        .collect();
    (0..n).for_each(|i| add(&mut heat, i, i, mass[i]));
    let solver = SparseSolver::from_pattern(n, heat.keys().copied().collect(), true);
    let mut delta = Vector::zero(n);
    delta[source] = 1.0;
    let u = solve(&solver, |i, j| heat[&(i, j)], &delta)?;
    let mut divergence = vec![0.0; n];
    for simplex in simplices {
        let ids = simplex.nodes.map(local);
        let gradient: [f64; D] =
            from_fn(|k| (0..N).map(|a| u[ids[a]] * simplex.gradients[a][k]).sum());
        let norm = dot(&gradient, &gradient).sqrt();
        if norm > 0.0 {
            for a in 0..N {
                divergence[ids[a]] -= simplex.volume * dot(&simplex.gradients[a], &gradient) / norm;
            }
        }
    }
    let mut b: Vector = divergence.into_iter().collect();
    b[source] = 0.0;
    let phi = solve(
        &solver,
        |i, j| {
            if i == source || j == source {
                f64::from(i == j)
            } else {
                stiffness[&(i, j)]
            }
        },
        &b,
    )?;
    let mut distances: Vec<f64> = phi.iter().copied().collect();
    distances[source] = 0.0;
    let minimum = distances.iter().copied().fold(f64::INFINITY, f64::min);
    distances.iter_mut().for_each(|d| *d -= minimum);
    Ok(nodes
        .into_iter()
        .zip(distances.into_iter().map(Quantity::new))
        .collect())
}

/// As [`Mesh::geodesic_distances_over`], from simplices already built for every element.
pub(crate) fn geodesic_distances_among<const D: usize, const N: usize>(
    mesh: &Mesh<D>,
    source: usize,
    simplices: &[Simplex<D, N>],
    elements: &[usize],
) -> Result<Vec<(usize, Quantity<Length>)>, &'static str> {
    let among: Vec<Simplex<D, N>> = elements.iter().map(|&element| simplices[element]).collect();
    heat(mesh, source, &among)
}

impl<const D: usize> Mesh<D> {
    /// Approximate geodesic distances from a source node to every node,
    /// through a mesh of triangles or tetrahedra, by the heat method.
    pub fn geodesic_distances(&self, source: usize) -> Result<Vec<Quantity<Length>>, &'static str> {
        let elements: Vec<usize> = (0..self.number_of_elements()).collect();
        let mut distances = vec![Quantity::new(f64::INFINITY); self.number_of_nodes()];
        self.geodesic_distances_over(source, &elements)?
            .into_iter()
            .for_each(|(node, distance)| distances[node] = distance);
        Ok(distances)
    }
    /// As [`Mesh::geodesic_distances`], but through only a subset of elements,
    /// returning the distance to each of their nodes in ascending node order.
    pub(crate) fn geodesic_distances_over(
        &self,
        source: usize,
        elements: &[usize],
    ) -> Result<Vec<(usize, Quantity<Length>)>, &'static str> {
        if let Some(triangles) = self.simplices_over::<3>(elements) {
            heat(self, source, &triangles)
        } else if let Some(tetrahedra) = self.simplices_over::<4>(elements) {
            heat(self, source, &tetrahedra)
        } else {
            Err(NOT_SIMPLICIAL)
        }
    }
}
