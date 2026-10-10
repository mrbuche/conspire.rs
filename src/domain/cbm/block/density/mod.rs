#[cfg(test)]
mod test;

use super::{Block, node::Weighting};
use crate::{
    domain::NodalReferenceCoordinates,
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, Tensor},
    units::Density,
};

pub use crate::domain::density::{DensityField, NoDensity};

pub trait Densities {
    fn at(&self, node: usize) -> Quantity<Density>;
}

pub type NodalDensities = Vec<Quantity<Density>>;

impl Densities for NodalDensities {
    fn at(&self, node: usize) -> Quantity<Density> {
        self[node]
    }
}

impl Densities for Quantity<Density> {
    fn at(&self, _node: usize) -> Quantity<Density> {
        *self
    }
}

impl<C, R> Block<C, R> {
    pub fn with_density<S>(self, density: S) -> Block<C, S> {
        Block {
            constitutive_model: self.constitutive_model,
            connectivity: self.connectivity,
            nodes: self.nodes,
            density,
        }
    }
    pub fn density(&self) -> &R {
        &self.density
    }
}

impl<C, D>
    From<(
        C,
        D,
        PrimitiveConnectivity<3, 4>,
        &NodalReferenceCoordinates<3>,
        Weighting,
    )> for Block<C, <D as DensityField>::Resolved<NodalDensities>>
where
    D: DensityField,
{
    fn from(
        (constitutive_model, density_field, connectivity, reference_coordinates, weighting): (
            C,
            D,
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
            Weighting,
        ),
    ) -> Self {
        let density = density_field.resolve(|field| {
            reference_coordinates
                .iter()
                .map(|coordinate| field.density(coordinate))
                .collect::<NodalDensities>()
        });
        Block::<C>::from((
            constitutive_model,
            connectivity,
            reference_coordinates,
            weighting,
        ))
        .with_density(density)
    }
}

impl<C, D>
    From<(
        C,
        D,
        PrimitiveConnectivity<3, 4>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C, <D as DensityField>::Resolved<NodalDensities>>
where
    D: DensityField,
{
    fn from(
        (constitutive_model, density_field, connectivity, reference_coordinates): (
            C,
            D,
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        (
            constitutive_model,
            density_field,
            connectivity,
            reference_coordinates,
            Weighting::default(),
        )
            .into()
    }
}
