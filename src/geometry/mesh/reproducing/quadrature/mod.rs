#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Basis, Mesh},
    math::Quantity,
    units::{Area, Volume},
};

const NOT_SIMPLICIAL: &str = "quadrature weights require a triangular or tetrahedral mesh";

/// The integral of each basis function over the mesh, in the coordinate unit
/// raised to the dimension.
///
/// The basis is linear over each element, so the integral is a sum over the
/// nodes of the value there times the share of the mesh assigned to the node.
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
    /// The integral of each basis function over the mesh, exact for its linear
    /// interpolation over the elements.
    ///
    /// These are the weights of the quadrature scheme the basis induces, which
    /// integrates fields the basis reproduces exactly.
    pub fn integrals(&self, basis: &Basis) -> Result<Vec<Quantity<Area>>, &'static str> {
        Ok(integrals(self, basis)?
            .into_iter()
            .map(Quantity::new)
            .collect())
    }
}

impl Mesh<3> {
    /// The integral of each basis function over the mesh, exact for its linear
    /// interpolation over the elements.
    ///
    /// These are the weights of the quadrature scheme the basis induces, which
    /// integrates fields the basis reproduces exactly.
    pub fn integrals(&self, basis: &Basis) -> Result<Vec<Quantity<Volume>>, &'static str> {
        Ok(integrals(self, basis)?
            .into_iter()
            .map(Quantity::new)
            .collect())
    }
}
