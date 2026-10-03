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
    math::{Quantity, Tensor, TensorList, TensorListVec},
    units::{Density, Mass},
};

pub use crate::domain::density::{DensityField, NoDensity};

pub trait Densities<const G: usize> {
    fn at(&self, element: usize) -> TensorList<Quantity<Density>, G>;
}

pub type ElementDensities<const G: usize> = TensorListVec<Quantity<Density>, G>;

impl<const G: usize> Densities<G> for ElementDensities<G> {
    fn at(&self, element: usize) -> TensorList<Quantity<Density>, G> {
        self[element].clone()
    }
}

impl<const G: usize> Densities<G> for Quantity<Density> {
    fn at(&self, _element: usize) -> TensorList<Quantity<Density>, G> {
        [*self; G].into()
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

impl<C, F, R, const G: usize, const N: usize> Block<C, F, G, 3, N, N, R>
where
    F: FiniteElement<G, 3, N, N>,
    R: Densities<G>,
{
    pub fn mass(&self) -> Quantity<Mass> {
        self.elements()
            .iter()
            .enumerate()
            .flat_map(|(element_index, element)| {
                self.density()
                    .at(element_index)
                    .into_iter()
                    .zip(element.integration_weights())
                    .map(|(density, integration_weight)| density * integration_weight)
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
    )> for Block<C, F, G, 3, N, N, <D as DensityField>::Resolved<ElementDensities<G>>>
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
        let density = density_field.resolve(|field| {
            connectivity
                .iter()
                .map(|nodes| {
                    let element_coordinates = Self::element_coordinates(coordinates, nodes);
                    shape_functions
                        .iter()
                        .map(|shape_functions| {
                            field.density(
                                &element_coordinates
                                    .iter()
                                    .zip(shape_functions.iter())
                                    .map(|(coordinate, shape_function)| coordinate * shape_function)
                                    .sum(),
                            )
                        })
                        .collect()
                })
                .collect::<ElementDensities<G>>()
        });
        Block::from((constitutive_model, connectivity, coordinates)).with_density(density)
    }
}

impl<C, F, D, const G: usize, const N: usize>
    From<(C, D, Vec<[usize; N]>, &NodalReferenceCoordinates<3>)>
    for Block<C, F, G, 3, N, N, <D as DensityField>::Resolved<ElementDensities<G>>>
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
