#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Basis, Mesh, simplex::Simplex},
    math::{FxHashMap, Reference, TensorRank1},
    units::ReciprocalLength,
};

pub type GradientVector<const D: usize> = TensorRank1<D, Reference, ReciprocalLength>;

type Entries<const D: usize> = Vec<(usize, [f64; D])>;

const NOT_SIMPLICIAL: &str = "projected gradients require a triangular or tetrahedral mesh";

/// The gradient of each approximation function at each quadrature function.
///
/// For each quadrature function, the index of each approximation function that
/// is not zero on its support, in ascending order, and the gradient there.
#[derive(Clone, Debug)]
pub struct Gradients<const D: usize> {
    pub values: Vec<Vec<(usize, GradientVector<D>)>>,
}

fn by_node(basis: &Basis, nodes: usize) -> Vec<Vec<(usize, f64)>> {
    let mut by_node = vec![Vec::new(); nodes];
    for (index, values) in basis.values.iter().enumerate() {
        values
            .iter()
            .for_each(|&(node, value)| by_node[node].push((index, value)));
    }
    by_node
}

fn project<const D: usize, const N: usize>(
    mesh: &Mesh<D>,
    approximation: &Basis,
    quadrature: &Basis,
    simplices: &[Simplex<D, N>],
) -> Result<Vec<Entries<D>>, &'static str> {
    let approximation = by_node(approximation, mesh.number_of_nodes());
    let quadrature_nodes = by_node(quadrature, mesh.number_of_nodes());
    let count = quadrature.values.len();
    let mut sums: Vec<FxHashMap<usize, [f64; D]>> = vec![FxHashMap::default(); count];
    let mut weights = vec![0.0; count];
    for simplex in simplices {
        let mut gradients: FxHashMap<usize, [f64; D]> = FxHashMap::default();
        for (a, &node) in simplex.nodes.iter().enumerate() {
            for &(function, value) in &approximation[node] {
                let gradient = gradients.entry(function).or_insert([0.0; D]);
                (0..D).for_each(|c| gradient[c] += value * simplex.gradients[a][c]);
            }
        }
        let mut measures: FxHashMap<usize, f64> = FxHashMap::default();
        for &node in &simplex.nodes {
            for &(function, value) in &quadrature_nodes[node] {
                *measures.entry(function).or_insert(0.0) += value * simplex.volume / N as f64;
            }
        }
        for (&point, &measure) in &measures {
            weights[point] += measure;
            for (&function, gradient) in &gradients {
                let sum = sums[point].entry(function).or_insert([0.0; D]);
                (0..D).for_each(|c| sum[c] += measure * gradient[c]);
            }
        }
    }
    if weights.iter().any(|&weight| weight <= 0.0) {
        return Err("a quadrature function has no positive integral");
    }
    Ok(sums
        .into_iter()
        .zip(weights)
        .map(|(sum, weight)| {
            let mut entries: Vec<(usize, [f64; D])> = sum
                .into_iter()
                .map(|(function, gradient)| (function, gradient.map(|g| g / weight)))
                .collect();
            entries.sort_unstable_by_key(|&(function, _)| function);
            entries
        })
        .collect())
}

impl<const D: usize> Mesh<D> {
    /// The lumped projected gradients of the approximation functions onto the
    /// quadrature functions.
    ///
    /// The gradient of each approximation function at a quadrature function is
    /// the average of its gradient over the mesh, weighted by the quadrature
    /// function. Reproducing linear fields, they give each quadrature function
    /// the exact gradient of a linear field, and integrate to the gradient of
    /// each approximation function.
    pub fn projected_gradients(
        &self,
        approximation: &Basis,
        quadrature: &Basis,
    ) -> Result<Gradients<D>, &'static str> {
        let elements: Vec<usize> = (0..self.number_of_elements()).collect();
        let values = if let Some(triangles) = self.simplices_over::<3>(&elements) {
            project(self, approximation, quadrature, &triangles)?
        } else if let Some(tetrahedra) = self.simplices_over::<4>(&elements) {
            project(self, approximation, quadrature, &tetrahedra)?
        } else {
            return Err(NOT_SIMPLICIAL);
        };
        Ok(Gradients {
            values: values
                .into_iter()
                .map(|entries| {
                    entries
                        .into_iter()
                        .map(|(function, gradient)| (function, GradientVector::from(gradient)))
                        .collect()
                })
                .collect(),
        })
    }
}
