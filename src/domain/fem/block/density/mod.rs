#[cfg(test)]
mod test;

use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{
            Block,
            element::{ElementNodalReferenceCoordinates, FiniteElement},
        },
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, Tensor, TensorList},
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
};

#[derive(Clone, Copy, Debug)]
pub struct NoDensity;

pub trait DensityField {
    fn density(&self, coordinate: &ReferenceCoordinate) -> Quantity<Density>;
}

impl DensityField for Quantity<Density> {
    fn density(&self, _coordinate: &ReferenceCoordinate) -> Quantity<Density> {
        *self
    }
}

impl<F> DensityField for F
where
    F: Fn(&ReferenceCoordinate) -> Quantity<Density>,
{
    fn density(&self, coordinate: &ReferenceCoordinate) -> Quantity<Density> {
        self(coordinate)
    }
}

#[derive(Clone, Debug)]
pub struct ElementDensities<const G: usize>(Vec<TensorList<Quantity<Density>, G>>);

impl<const G: usize> ElementDensities<G> {
    pub fn iter(&self) -> impl Iterator<Item = &TensorList<Quantity<Density>, G>> {
        self.0.iter()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R>
    Block<C, F, G, M, N, P, R>
{
    pub fn with_density<S>(self, density: S) -> Block<C, F, G, M, N, P, S> {
        Block {
            constitutive_model: self.constitutive_model,
            connectivity: self.connectivity,
            elements: self.elements,
            density,
        }
    }
    pub fn density(&self) -> &R {
        &self.density
    }
}

impl<C, F, const G: usize, const N: usize> Block<C, F, G, 3, N, N, ElementDensities<G>>
where
    F: FiniteElement<G, 3, N, N>,
{
    pub fn mass(&self) -> Quantity<Mass> {
        self.density()
            .iter()
            .zip(self.elements())
            .map(|(densities, element)| {
                densities
                    .iter()
                    .zip(element.integration_weights())
                    .map(|(density, integration_weight)| density * integration_weight)
                    .sum::<Quantity<Mass>>()
            })
            .sum()
    }
}

impl<C, F, D, const G: usize, const N: usize>
    From<(
        C,
        D,
        PrimitiveConnectivity<3, N>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C, F, G, 3, N, N, ElementDensities<G>>
where
    F: FiniteElement<G, 3, N, N> + From<ElementNodalReferenceCoordinates<N>>,
    D: DensityField,
{
    fn from(
        (constitutive_model, density_field, connectivity, coordinates): (
            C,
            D,
            PrimitiveConnectivity<3, N>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let shape_functions = F::shape_functions_at_integration_points();
        let densities = connectivity
            .iter()
            .map(|nodes| {
                let element_coordinates = Self::element_coordinates(coordinates, nodes);
                shape_functions
                    .iter()
                    .map(|shape_functions| {
                        density_field.density(
                            &element_coordinates
                                .iter()
                                .zip(shape_functions.iter())
                                .map(|(coordinate, shape_function)| coordinate * shape_function)
                                .sum::<ReferenceCoordinate>(),
                        )
                    })
                    .collect()
            })
            .collect();
        Block::<C, F, G, 3, N, N>::from((constitutive_model, connectivity, coordinates))
            .with_density(ElementDensities(densities))
    }
}

impl<C, F, D, const G: usize, const N: usize>
    From<(C, D, Vec<[usize; N]>, &NodalReferenceCoordinates<3>)>
    for Block<C, F, G, 3, N, N, ElementDensities<G>>
where
    F: FiniteElement<G, 3, N, N> + From<ElementNodalReferenceCoordinates<N>>,
    D: DensityField,
{
    fn from(
        (constitutive_model, density_field, connectivity, coordinates): (
            C,
            D,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        Self::from((
            constitutive_model,
            density_field,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        ))
    }
}
