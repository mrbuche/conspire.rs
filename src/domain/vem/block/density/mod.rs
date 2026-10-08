use crate::{
    geometry::mesh::PolytopalConnectivity,
    math::{Quantity, Scalar, Tensor},
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
    vem::{
        NodalReferenceCoordinates,
        block::{Block, element::VirtualElement},
    },
};

pub use crate::domain::density::{DensityField, NoDensity};

pub trait Densities {
    fn at(&self, element: usize) -> Quantity<Density>;
}

pub type ElementDensities = Vec<Quantity<Density>>;

impl Densities for ElementDensities {
    fn at(&self, element: usize) -> Quantity<Density> {
        self[element]
    }
}

impl Densities for Quantity<Density> {
    fn at(&self, _element: usize) -> Quantity<Density> {
        *self
    }
}

impl<C, F, R> Block<C, F, R> {
    pub fn with_density<S>(self, density: S) -> Block<C, F, S> {
        Block {
            constitutive_model: self.constitutive_model,
            connectivity: self.connectivity,
            elements: self.elements,
            elements_nodes: self.elements_nodes,
            density,
        }
    }
    pub fn density(&self) -> &R {
        &self.density
    }
}

impl<C, F, R> Block<C, F, R>
where
    F: VirtualElement,
    R: Densities,
{
    pub fn mass(&self) -> Quantity<Mass> {
        self.elements()
            .iter()
            .enumerate()
            .map(|(element_index, element)| {
                self.density().at(element_index) * element.integration_weights()[0]
            })
            .sum()
    }
}

impl<C, F, D>
    From<(
        C,
        D,
        PolytopalConnectivity<3>,
        &NodalReferenceCoordinates,
        Scalar,
    )> for Block<C, F, <D as DensityField>::Resolved<ElementDensities>>
where
    F: VirtualElement,
    D: DensityField,
{
    fn from(
        (constitutive_model, density_field, connectivity, coordinates, stabilization): (
            C,
            D,
            PolytopalConnectivity<3>,
            &NodalReferenceCoordinates,
            Scalar,
        ),
    ) -> Self {
        let block =
            Block::<C, F>::from((constitutive_model, connectivity, coordinates, stabilization));
        let density = density_field.resolve(|field| {
            block
                .elements_nodes()
                .iter()
                .map(|nodes| {
                    field.density(
                        &(Block::<C, F>::element_coordinates(coordinates, nodes)
                            .iter()
                            .sum::<ReferenceCoordinate>()
                            / (nodes.len() as Scalar)),
                    )
                })
                .collect::<ElementDensities>()
        });
        block.with_density(density)
    }
}

impl<C, F, D>
    From<(
        C,
        D,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        &NodalReferenceCoordinates,
        Scalar,
    )> for Block<C, F, <D as DensityField>::Resolved<ElementDensities>>
where
    F: VirtualElement,
    D: DensityField,
{
    fn from(
        (
            constitutive_model,
            density_field,
            elements_faces,
            faces_nodes,
            coordinates,
            stabilization,
        ): (
            C,
            D,
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
            &NodalReferenceCoordinates,
            Scalar,
        ),
    ) -> Self {
        Self::from((
            constitutive_model,
            density_field,
            PolytopalConnectivity::from((elements_faces, faces_nodes)),
            coordinates,
            stabilization,
        ))
    }
}
