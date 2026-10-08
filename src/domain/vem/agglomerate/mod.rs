#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::block::element::solid::elastic::ElasticElement,
    geometry::mesh::{
        Mesh,
        partition::agglomerate::{outward_faces, union_faces},
    },
    math::{Quantity, Scalar},
    units::{Density, Stress, Time},
    vem::{
        NodalReferenceCoordinates,
        block::element::{
            Element, ElementNodalReferenceCoordinates,
            mass::{ElementNodalLumpedMasses, LumpedMassVirtualElement},
            solid::ElementNodalStiffnessesSolid,
            time_scale::{fastest_time_scale, time_scale_exceeds},
        },
    },
};

/// The time scale of any union of the elements of a mesh as one virtual element.
///
/// A unit elastic material and a unit density are used, since only ratios of time scales matter.
pub struct Candidates {
    elements_faces: Vec<Vec<Vec<usize>>>,
    coordinates: NodalReferenceCoordinates,
    material: NeoHookean,
    stabilization: Scalar,
}

impl Candidates {
    pub fn new(mesh: &Mesh<3>, poisson: Scalar, stabilization: Scalar) -> Result<Self, String> {
        Ok(Self {
            elements_faces: outward_faces(mesh)?,
            coordinates: mesh.coordinates().clone(),
            material: NeoHookean {
                bulk_modulus: Stress::pascals(1.0 / (3.0 * (1.0 - 2.0 * poisson))),
                shear_modulus: Stress::pascals(0.5 / (1.0 + poisson)),
            },
            stabilization,
        })
    }
    pub fn number_of_elements(&self) -> usize {
        self.elements_faces.len()
    }
    pub fn time_scale(&self, elements: &[usize]) -> Result<Quantity<Time>, String> {
        let (stiffnesses, masses) = self.matrices(elements)?;
        Ok(fastest_time_scale(&stiffnesses, &masses))
    }
    pub fn time_scales(&self) -> Result<Vec<Quantity<Time>>, String> {
        (0..self.number_of_elements())
            .map(|element| self.time_scale(&[element]))
            .collect()
    }
    pub fn time_scale_exceeds(
        &self,
        elements: &[usize],
        minimum: Quantity<Time>,
    ) -> Result<bool, String> {
        let (stiffnesses, masses) = self.matrices(elements)?;
        Ok(time_scale_exceeds(&stiffnesses, &masses, minimum))
    }
    fn matrices(
        &self,
        elements: &[usize],
    ) -> Result<(ElementNodalStiffnessesSolid, ElementNodalLumpedMasses), String> {
        let faces = union_faces(&self.elements_faces, elements)?;
        let mut nodes = faces.iter().flatten().copied().collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        let faces_coordinates: ElementNodalReferenceCoordinates = faces
            .iter()
            .map(|face| {
                face.iter()
                    .map(|&node| self.coordinates[node].clone())
                    .collect()
            })
            .collect();
        let element = Element::from((
            faces_coordinates,
            &(0..faces.len()).collect::<Vec<_>>()[..],
            &nodes[..],
            &faces[..],
            self.stabilization,
        ));
        let coordinates = nodes
            .iter()
            .map(|&node| self.coordinates[node].clone())
            .collect::<NodalReferenceCoordinates>();
        let stiffnesses = element
            .nodal_stiffnesses(&self.material, &coordinates.clone().into())
            .map_err(|error| format!("{error:?}"))?;
        let masses =
            element.nodal_lumped_masses(Density::kilograms_per_cubic_meter(1.0), &coordinates);
        Ok((stiffnesses, masses))
    }
}
