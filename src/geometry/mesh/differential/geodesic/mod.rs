#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, simplex::dot},
    math::{Quantity, Tensor, Vector, sparse::SparseSolver},
    units::Length,
};
use std::{array::from_fn, collections::HashMap};

type Entries = HashMap<(usize, usize), f64>;

fn add(entries: &mut Entries, i: usize, j: usize, value: f64) {
    *entries.entry((i, j)).or_insert(0.0) += value;
}

fn solve(entries: &Entries, b: &Vector) -> Result<Vector, &'static str> {
    let pattern: Vec<(usize, usize)> = entries.keys().copied().collect();
    SparseSolver::from_pattern(b.len(), pattern, true)
        .solve(|i, j| entries[&(i, j)], b)
        .map_err(|_| "geodesic linear solve failed")
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
        let simplices = self
            .simplices_over(elements)
            .ok_or("geodesic distances require a triangular or tetrahedral mesh")?;
        let coordinates = self.coordinates();
        let point =
            |node: usize| -> [f64; D] { from_fn(|k| coordinates[node][k].value_as::<Length>()) };
        let mut nodes: Vec<usize> = simplices
            .iter()
            .flat_map(|s| s.nodes.iter().copied())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        let source = nodes
            .binary_search(&source)
            .map_err(|_| "source node is not in the elements")?;
        let local = |node: usize| nodes.binary_search(&node).expect("node in elements");
        let n = nodes.len();
        let mut stiffness = Entries::new();
        let mut mass = vec![0.0; n];
        let (mut length, mut count) = (0.0, 0);
        for simplex in &simplices {
            let ids: Vec<usize> = simplex.nodes.iter().map(|&node| local(node)).collect();
            let m = ids.len();
            for a in 0..m {
                mass[ids[a]] += simplex.volume / m as f64;
                for b in a + 1..m {
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
        let mut delta = Vector::zero(n);
        delta[source] = 1.0;
        let u = solve(&heat, &delta)?;
        let mut divergence = vec![0.0; n];
        for simplex in &simplices {
            let ids: Vec<usize> = simplex.nodes.iter().map(|&node| local(node)).collect();
            let gradient: [f64; D] = from_fn(|k| {
                ids.iter()
                    .zip(&simplex.gradients)
                    .map(|(&id, g)| u[id] * g[k])
                    .sum()
            });
            let norm = dot(&gradient, &gradient).sqrt();
            if norm > 0.0 {
                ids.iter().zip(&simplex.gradients).for_each(|(&id, g)| {
                    divergence[id] -= simplex.volume * dot(g, &gradient) / norm
                });
            }
        }
        let reduced = |i: usize| if i > source { i - 1 } else { i };
        let poisson: Entries = stiffness
            .iter()
            .filter(|&(&(i, j), _)| i != source && j != source)
            .map(|(&(i, j), &value)| ((reduced(i), reduced(j)), value))
            .collect();
        let b: Vector = (0..n)
            .filter(|&i| i != source)
            .map(|i| divergence[i])
            .collect();
        let phi = solve(&poisson, &b)?;
        let mut distances: Vec<f64> = (0..n)
            .map(|i| if i == source { 0.0 } else { phi[reduced(i)] })
            .collect();
        let minimum = distances.iter().copied().fold(f64::INFINITY, f64::min);
        distances.iter_mut().for_each(|d| *d -= minimum);
        Ok(nodes
            .into_iter()
            .zip(distances.into_iter().map(Quantity::new))
            .collect())
    }
}
