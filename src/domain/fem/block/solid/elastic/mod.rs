pub mod internal_variables;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::solid::elastic::ElasticItems,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError,
                planar::PlanarElasticFiniteElement,
                solid::{
                    ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                    elastic::ElasticFiniteElement,
                },
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
};
#[cfg(feature = "gpu")]
use crate::{
    constitutive::solid::elastic::GpuModel,
    domain::{
        gpu::{GpuItems, LibraryBlock, ShapeTables},
        parallel::ItemIndex,
    },
    fem::block::element::{ElementNodalCoordinates, linear::Hexahedron},
    math::{Quantity, Tensor},
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> ElasticItems<3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
{
    type Forces = ElementNodalForcesSolid<N>;
    type Stiffnesses = ElementNodalStiffnessesSolid<N>;
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementNodalForcesSolid<N>, ElementModelError> {
        self.elements()[item]
            .nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(
                    nodal_coordinates,
                    &self.connectivity().iter().as_slice()[item],
                ),
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, ElementModelError> {
        self.elements()[item]
            .nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(
                    nodal_coordinates,
                    &self.connectivity().iter().as_slice()[item],
                ),
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    /// The block as a device takes it, for the elements with three dimensions.
    ///
    /// The deformation gradient of such an element is the sum over its nodes of the outer
    /// product of the coordinates and the gradient vector of the node, so the gradient vectors
    /// are those of the deformation gradients when the coordinates are zero but for a unit
    /// first component at one node.
    #[cfg(feature = "gpu")]
    fn gpu_block(&self) -> Option<GpuItems> {
        if M != 3 {
            return None;
        }
        let model = self.constitutive_model().gpu_model()?;
        let mut reference = Vec::with_capacity(self.elements().len() * (G * N * 3 + G));
        for element in self.elements() {
            let mut gradients = vec![0.0; G * N * 3];
            for node in 0..N {
                let mut coordinates = ElementNodalCoordinates::<N>::default();
                coordinates[node][0] = Quantity::new(1.0);
                element
                    .deformation_gradients(&coordinates)
                    .iter()
                    .enumerate()
                    .for_each(|(point, deformation_gradient)| {
                        (0..3).for_each(|component| {
                            gradients[(point * N + node) * 3 + component] =
                                deformation_gradient[0][component].value()
                        })
                    });
            }
            reference.extend(gradients);
            reference.extend(
                element
                    .integration_weights()
                    .iter()
                    .map(|weight| weight.value()),
            );
        }
        let (offsets, contributions) = ItemIndex::new(self).flattened(N);
        // An element makes its gradient vectors from the derivatives of its shape functions at
        // its integration points when it is made, if it has a shape function for each node.
        let shape = (P == N).then(|| {
            let derivatives = F::shape_functions_gradients_at_integration_points();
            ShapeTables {
                derivatives: derivatives
                    .iter()
                    .flat_map(|point| {
                        point
                            .iter()
                            .flat_map(|node| node.iter().map(|component| component.value()))
                    })
                    .collect(),
                weights: F::parametric_weights().iter().copied().collect(),
            }
        });
        // The kernels of the library are for the elements that they were built for, which are
        // known by their types, and their elements are the bytes that they are.
        let library = (core::any::type_name::<F>() == core::any::type_name::<Hexahedron>()
            && matches!(model, GpuModel::NeoHookean { .. }))
        .then(|| {
            let elements = self.elements();
            // SAFETY: the elements are plain numbers, which the device reads as bytes.
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    elements.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(elements),
                )
            };
            LibraryBlock {
                elements: bytes.to_vec(),
                forces_kernel: "block_forces",
            }
        });
        Some(GpuItems {
            library,
            shape,
            model,
            nodes_per_item: N,
            points_per_item: G,
            connectivity: self
                .connectivity()
                .iter()
                .flat_map(|nodes| nodes.iter().map(|&node| node as u32))
                .collect(),
            reference,
            offsets,
            contributions,
        })
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize, R> ElasticElements<2>
    for Block<C, F, G, 2, N, P, R>
where
    C: Elastic,
    F: PlanarElasticFiniteElement<C, G, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_forces: &mut NodalForcesSolid<2>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<2>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
