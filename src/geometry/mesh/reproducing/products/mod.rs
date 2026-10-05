#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Basis, Mesh, simplex::Simplex},
    math::{FxHashMap, Quantity},
    units::{Area, Volume},
};

const NOT_SIMPLICIAL: &str = "inner products require a triangular or tetrahedral mesh";

/// For each function, the index of each function that shares support with it, in
/// ascending order, and the integral of their product.
pub type InnerProducts<U> = Vec<Vec<(usize, Quantity<U>)>>;

type Rows = Vec<Vec<(usize, f64)>>;

fn by_node(basis: &Basis, nodes: usize) -> Vec<Vec<(usize, f64)>> {
    let mut by_node = vec![Vec::new(); nodes];
    for (index, values) in basis.values.iter().enumerate() {
        values
            .iter()
            .for_each(|&(node, value)| by_node[node].push((index, value)));
    }
    by_node
}

fn products<const D: usize, const N: usize>(
    mesh: &Mesh<D>,
    basis: &Basis,
    simplices: &[Simplex<D, N>],
) -> Rows {
    let by_node = by_node(basis, mesh.number_of_nodes());
    let mut rows = vec![FxHashMap::default(); basis.values.len()];
    for simplex in simplices {
        let mut local = FxHashMap::<usize, [f64; N]>::default();
        for (a, &node) in simplex.nodes.iter().enumerate() {
            for &(function, value) in &by_node[node] {
                local.entry(function).or_insert([0.0; N])[a] = value;
            }
        }
        let scale = simplex.volume / (N * (N + 1)) as f64;
        let sums: Vec<(usize, f64, [f64; N])> = local
            .into_iter()
            .map(|(function, values)| (function, values.iter().sum(), values))
            .collect();
        for &(i, sum_i, values_i) in &sums {
            for &(j, sum_j, values_j) in &sums {
                let diagonal: f64 = (0..N).map(|a| values_i[a] * values_j[a]).sum();
                *rows[i].entry(j).or_insert(0.0) += scale * (sum_i * sum_j + diagonal);
            }
        }
    }
    rows.into_iter()
        .map(|row| {
            let mut row: Vec<(usize, f64)> = row.into_iter().collect();
            row.sort_unstable_by_key(|&(function, _)| function);
            row
        })
        .collect()
}

fn inner_products<const D: usize>(mesh: &Mesh<D>, basis: &Basis) -> Result<Rows, &'static str> {
    let elements: Vec<usize> = (0..mesh.number_of_elements()).collect();
    if let Some(triangles) = mesh.simplices_over::<3>(&elements) {
        Ok(products(mesh, basis, &triangles))
    } else if let Some(tetrahedra) = mesh.simplices_over::<4>(&elements) {
        Ok(products(mesh, basis, &tetrahedra))
    } else {
        Err(NOT_SIMPLICIAL)
    }
}

impl Mesh<2> {
    /// The integral over the mesh of the product of each pair of functions
    /// with overlapping support, each taken as linear over every element.
    pub fn inner_products(&self, basis: &Basis) -> Result<InnerProducts<Area>, &'static str> {
        Ok(inner_products(self, basis)?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(function, product)| (function, Quantity::new(product)))
                    .collect()
            })
            .collect())
    }
}

impl Mesh<3> {
    /// The integral over the mesh of the product of each pair of functions
    /// with overlapping support, each taken as linear over every element.
    pub fn inner_products(&self, basis: &Basis) -> Result<InnerProducts<Volume>, &'static str> {
        Ok(inner_products(self, basis)?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(function, product)| (function, Quantity::new(product)))
                    .collect()
            })
            .collect())
    }
}
