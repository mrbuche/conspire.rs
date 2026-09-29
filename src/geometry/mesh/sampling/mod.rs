#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::Mesh,
    math::{Tensor, random::Rng},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet},
};

struct Packing<const D: usize> {
    spacing: f64,
    cells: HashMap<[i64; D], Vec<[f64; D]>>,
}

impl<const D: usize> Packing<D> {
    fn key(&self, point: &[f64; D]) -> [i64; D] {
        from_fn(|k| (point[k] / self.spacing).floor() as i64)
    }
    fn is_free(&self, point: &[f64; D]) -> bool {
        let key = self.key(point);
        (0..3usize.pow(D as u32)).all(|offset| {
            let neighbor: [i64; D] =
                from_fn(|k| key[k] + (offset / 3usize.pow(k as u32) % 3) as i64 - 1);
            self.cells.get(&neighbor).is_none_or(|others| {
                others.iter().all(|other| {
                    let squared: f64 = (0..D).map(|k| (point[k] - other[k]).powi(2)).sum();
                    squared >= self.spacing * self.spacing
                })
            })
        })
    }
    fn insert(&mut self, point: [f64; D]) {
        self.cells.entry(self.key(&point)).or_default().push(point);
    }
}

impl<const D: usize> Mesh<D> {
    /// Nodes forming a random close packing with hard-sphere diameter `spacing`.
    ///
    /// Boundary nodes are sampled first, then the remaining nodes. Every node
    /// is left within `spacing` of a sampled node, and the boundary nodes are
    /// left within `spacing` of a sampled boundary node. The `seed` makes the
    /// sampling deterministic.
    pub fn sample(&self, spacing: f64, seed: u64) -> Vec<usize> {
        assert!(spacing > 0.0, "Sampling spacing must be positive.");
        let points: Vec<[f64; D]> = self
            .coordinates()
            .iter()
            .map(|x| std::array::from_fn(|k| x[k].value()))
            .collect();
        let on_boundary: HashSet<usize> = self.exterior_faces().into_iter().flatten().collect();
        let mut boundary: Vec<usize> = on_boundary.iter().copied().collect();
        let mut interior: Vec<usize> = (0..self.number_of_nodes())
            .filter(|&node| {
                !self.node_element_connectivity()[node].is_empty() && !on_boundary.contains(&node)
            })
            .collect();
        boundary.sort_unstable();
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut boundary);
        rng.shuffle(&mut interior);
        let mut packing = Packing {
            spacing,
            cells: HashMap::new(),
        };
        boundary
            .into_iter()
            .chain(interior)
            .filter(|&node| {
                let free = packing.is_free(&points[node]);
                if free {
                    packing.insert(points[node]);
                }
                free
            })
            .collect()
    }
}
