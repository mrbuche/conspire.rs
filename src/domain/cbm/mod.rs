//! Continuum Bond Method (CBM) kinematics.
//!
//! Sperling, Hoefnagels, van den Broek, Geers, "A continuum consistent
//! discrete particle method for continuum-discontinuum transitions and
//! complex fracture problems," CMAME 390 (2022) 114460.
//! <https://doi.org/10.1016/j.cma.2021.114460>
//!
//! Particle deformation gradients are volume-weighted averages of the
//! per-tetrahedron constant deformation gradients over the tetrahedra
//! incident to each particle (Eqs. 1-2 of the paper, generalized from
//! triangles to tetrahedra).

#[cfg(test)]
mod test;

use crate::{
    domain::{NodalCoordinates, NodalReferenceCoordinates, NodalVelocities},
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, linear::Tetrahedron, solid::SolidElement,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, Tensor, TensorArray},
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::Volume,
};

pub use crate::domain::solid::SolidElements;

pub struct Cbm {
    connectivity: PrimitiveConnectivity<3, 4>,
    elements: Vec<Tetrahedron>,
    node_volumes: Vec<Quantity<Volume>>,
}

impl Cbm {
    fn element_coordinates<const D: usize, I, U>(
        coordinates: &crate::math::TensorRank1Vec<D, I, U>,
        nodes: &[usize; 4],
    ) -> crate::math::TensorRank1List<D, I, 4, U> {
        nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect()
    }
}

impl From<(PrimitiveConnectivity<3, 4>, &NodalReferenceCoordinates<3>)> for Cbm {
    fn from(
        (connectivity, reference_coordinates): (
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let elements: Vec<Tetrahedron> = connectivity
            .iter()
            .map(|nodes| -> ElementNodalReferenceCoordinates<4> {
                Self::element_coordinates(reference_coordinates, nodes)
            })
            .map(Tetrahedron::from)
            .collect();
        let mut node_volumes = vec![Quantity::<Volume>::new(0.0); reference_coordinates.len()];
        connectivity
            .iter()
            .zip(elements.iter())
            .for_each(|(nodes, element)| {
                let quarter_volume = element.volume() / 4.0;
                nodes
                    .iter()
                    .for_each(|&node| node_volumes[node] += &quarter_volume)
            });
        Self {
            connectivity,
            elements,
            node_volumes,
        }
    }
}

impl SolidElements for Cbm {
    type DeformationGradients = DeformationGradient;
    type DeformationGradientRates = DeformationGradientRate;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients> {
        let mut particle_deformation_gradients =
            vec![DeformationGradient::zero(); self.node_volumes.len()];
        self.connectivity
            .iter()
            .zip(self.elements.iter())
            .for_each(|(nodes, element)| {
                let element_deformation_gradient = element
                    .deformation_gradients(&Self::element_coordinates(nodal_coordinates, nodes))[0]
                    .clone();
                let quarter_volume = element.volume() / 4.0;
                nodes.iter().for_each(|&node| {
                    let weight = (quarter_volume / self.node_volumes[node]).value();
                    particle_deformation_gradients[node] +=
                        element_deformation_gradient.clone() * weight
                })
            });
        particle_deformation_gradients
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        let mut particle_deformation_gradient_rates =
            vec![DeformationGradientRate::zero(); self.node_volumes.len()];
        self.connectivity
            .iter()
            .zip(self.elements.iter())
            .for_each(|(nodes, element)| {
                let element_deformation_gradient_rate = element.deformation_gradient_rates(
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )[0]
                .clone();
                let quarter_volume = element.volume() / 4.0;
                nodes.iter().for_each(|&node| {
                    let weight = (quarter_volume / self.node_volumes[node]).value();
                    particle_deformation_gradient_rates[node] +=
                        element_deformation_gradient_rate.clone() * weight
                })
            });
        particle_deformation_gradient_rates
    }
}
