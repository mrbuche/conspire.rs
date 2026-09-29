#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    math::{Tensor, Vector, sparse::SparseSolver},
};
use std::{array::from_fn, collections::HashMap};

type Entries = HashMap<(usize, usize), f64>;

struct Triangle<const D: usize> {
    nodes: [usize; 3],
    area: f64,
    gradients: [[f64; D]; 3],
}

impl<const D: usize> Triangle<D> {
    fn new(nodes: [usize; 3], points: [[f64; D]; 3]) -> Self {
        let u: [f64; D] = from_fn(|k| points[1][k] - points[0][k]);
        let v: [f64; D] = from_fn(|k| points[2][k] - points[0][k]);
        let dot = |a: &[f64; D], b: &[f64; D]| (0..D).map(|k| a[k] * b[k]).sum::<f64>();
        let (uu, uv, vv) = (dot(&u, &u), dot(&u, &v), dot(&v, &v));
        let determinant = uu * vv - uv * uv;
        let g1: [f64; D] = from_fn(|k| (vv * u[k] - uv * v[k]) / determinant);
        let g2: [f64; D] = from_fn(|k| (uu * v[k] - uv * u[k]) / determinant);
        Self {
            nodes,
            area: 0.5 * determinant.sqrt(),
            gradients: [from_fn(|k| -g1[k] - g2[k]), g1, g2],
        }
    }
}

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
    /// Approximate geodesic distances from a source node to every node, along
    /// the surface of an all-triangular mesh, by the heat method.
    ///
    /// Nodes that no element touches are infinitely far away.
    pub fn geodesic_distances(&self, source: usize) -> Result<Vec<f64>, &'static str> {
        let elements: Vec<usize> = (0..self.number_of_elements()).collect();
        let mut distances = vec![f64::INFINITY; self.number_of_nodes()];
        self.geodesic_distances_over(source, &elements)?
            .into_iter()
            .for_each(|(node, distance)| distances[node] = distance);
        Ok(distances)
    }
    /// As [`Mesh::geodesic_distances`], but along only a subset of elements,
    /// returning the distance to each of their nodes in ascending node order.
    pub(crate) fn geodesic_distances_over(
        &self,
        source: usize,
        elements: &[usize],
    ) -> Result<Vec<(usize, f64)>, &'static str> {
        if !self
            .iter()
            .all(|block| matches!(block, Connectivity::Triangular(_)))
        {
            return Err("geodesic distances require an all-triangular mesh");
        }
        let triangles: Vec<[usize; 3]> = self
            .iter()
            .flat_map(|block| {
                block
                    .iter()
                    .map(|element| [element[0], element[1], element[2]])
            })
            .collect();
        let coordinates = self.coordinates();
        let point = |node: usize| -> [f64; D] { from_fn(|k| coordinates[node][k].value()) };
        let triangles: Vec<Triangle<D>> = elements
            .iter()
            .map(|&element| {
                let nodes = triangles[element];
                Triangle::new(nodes, nodes.map(point))
            })
            .collect();
        let mut nodes: Vec<usize> = triangles.iter().flat_map(|t| t.nodes).collect();
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
        for triangle in &triangles {
            let ids = triangle.nodes.map(local);
            for a in 0..3 {
                let b = (a + 1) % 3;
                let weight: f64 = (0..D)
                    .map(|k| triangle.gradients[a][k] * triangle.gradients[b][k])
                    .sum::<f64>()
                    * triangle.area;
                add(&mut stiffness, ids[a], ids[b], weight);
                add(&mut stiffness, ids[b], ids[a], weight);
                add(&mut stiffness, ids[a], ids[a], -weight);
                add(&mut stiffness, ids[b], ids[b], -weight);
                length += (0..D)
                    .map(|k| {
                        (coordinates[triangle.nodes[a]][k].value()
                            - coordinates[triangle.nodes[b]][k].value())
                        .powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                count += 1;
                mass[ids[a]] += triangle.area / 3.0;
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
        for triangle in &triangles {
            let ids = triangle.nodes.map(local);
            let gradient: [f64; D] = from_fn(|k| {
                (0..3)
                    .map(|a| u[ids[a]] * triangle.gradients[a][k])
                    .sum::<f64>()
            });
            let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm > 0.0 {
                for a in 0..3 {
                    divergence[ids[a]] -= triangle.area
                        * (0..D)
                            .map(|k| triangle.gradients[a][k] * gradient[k])
                            .sum::<f64>()
                        / norm;
                }
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
        Ok(nodes.into_iter().zip(distances).collect())
    }
}
