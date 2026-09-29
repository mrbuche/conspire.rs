#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    units::Length,
};
use std::array::from_fn;

/// A simplex with N nodes embedded in D dimensions, with the constant
/// gradients of its linear shape functions.
pub(crate) struct Simplex<const D: usize, const N: usize> {
    pub(crate) nodes: [usize; N],
    pub(crate) volume: f64,
    pub(crate) gradients: [[f64; D]; N],
}

pub(crate) fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn invert(mut matrix: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, f64) {
    let n = matrix.len();
    let mut inverse: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut determinant = 1.0;
    for c in 0..n {
        let pivot = (c..n)
            .max_by(|&a, &b| matrix[a][c].abs().total_cmp(&matrix[b][c].abs()))
            .expect("nonempty matrix");
        if pivot != c {
            matrix.swap(pivot, c);
            inverse.swap(pivot, c);
            determinant = -determinant;
        }
        let diagonal = matrix[c][c];
        determinant *= diagonal;
        matrix[c].iter_mut().for_each(|x| *x /= diagonal);
        inverse[c].iter_mut().for_each(|x| *x /= diagonal);
        for r in (0..n).filter(|&r| r != c) {
            let factor = matrix[r][c];
            for j in 0..n {
                matrix[r][j] -= factor * matrix[c][j];
                inverse[r][j] -= factor * inverse[c][j];
            }
        }
    }
    (inverse, determinant)
}

impl<const D: usize, const N: usize> Simplex<D, N> {
    pub(crate) fn new(nodes: [usize; N], points: [[f64; D]; N]) -> Self {
        let k = N - 1;
        let edges: Vec<[f64; D]> = (1..N)
            .map(|i| from_fn(|c| points[i][c] - points[0][c]))
            .collect();
        let gram = (0..k)
            .map(|i| (0..k).map(|j| dot(&edges[i], &edges[j])).collect())
            .collect();
        let (inverse, determinant) = invert(gram);
        let last: Vec<[f64; D]> = (0..k)
            .map(|i| from_fn(|c| (0..k).map(|j| inverse[i][j] * edges[j][c]).sum()))
            .collect();
        let first: [f64; D] = from_fn(|c| -last.iter().map(|g| g[c]).sum::<f64>());
        Self {
            nodes,
            volume: determinant.sqrt() / (1..=k).product::<usize>() as f64,
            gradients: from_fn(|a| if a == 0 { first } else { last[a - 1] }),
        }
    }
}

impl<const D: usize> Mesh<D> {
    /// The given elements as simplices with N nodes, or nothing unless every
    /// block of the mesh is triangles (N = 3) or tetrahedra (N = 4).
    pub(crate) fn simplices_over<const N: usize>(
        &self,
        elements: &[usize],
    ) -> Option<Vec<Simplex<D, N>>> {
        if !self.iter().all(|block| {
            matches!(
                block,
                Connectivity::Triangular(_) | Connectivity::Tetrahedral(_)
            ) && block.number_of_nodes_per_element() == Some(N)
        }) {
            return None;
        }
        let all: Vec<&[usize]> = self.iter().flat_map(|block| block.iter()).collect();
        let coordinates = self.coordinates();
        let point =
            |node: usize| -> [f64; D] { from_fn(|k| coordinates[node][k].value_as::<Length>()) };
        Some(
            elements
                .iter()
                .map(|&element| {
                    let nodes: [usize; N] = from_fn(|c| all[element][c]);
                    Simplex::new(nodes, nodes.map(point))
                })
                .collect(),
        )
    }
}
