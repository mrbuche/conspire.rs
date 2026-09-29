#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Basis, Mesh},
    math::Quantity,
    units::{Area, Volume},
};

const NOT_SIMPLICIAL: &str = "quadrature weights require a triangular or tetrahedral mesh";

fn integrals<const D: usize>(mesh: &Mesh<D>, basis: &Basis) -> Result<Vec<f64>, &'static str> {
    let shares = mesh.node_shares().ok_or(NOT_SIMPLICIAL)?;
    Ok(basis
        .values
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&(node, value)| value * shares[node])
                .sum()
        })
        .collect())
}

impl Mesh<2> {
    pub fn integrals(&self, basis: &Basis) -> Result<Vec<Quantity<Area>>, &'static str> {
        Ok(integrals(self, basis)?
            .into_iter()
            .map(Quantity::new)
            .collect())
    }
}

impl Mesh<3> {
    pub fn integrals(&self, basis: &Basis) -> Result<Vec<Quantity<Volume>>, &'static str> {
        Ok(integrals(self, basis)?
            .into_iter()
            .map(Quantity::new)
            .collect())
    }
}
