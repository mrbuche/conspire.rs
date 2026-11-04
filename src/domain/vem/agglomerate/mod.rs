mod finite;
mod merge;

pub use merge::{Agglomerated, Agglomeration, Reference};
#[cfg(test)]
mod test;

use self::finite::Primitive;
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::{block::element::solid::elastic::ElasticElement, fem::block::element::FiniteElement},
    geometry::mesh::{Boundary, ElementsFaces, Mesh},
    math::{Quantity, Scalar},
    units::{Density, Stress, Time},
    vem::{
        NodalReferenceCoordinates,
        block::element::{
            Element, ElementNodalReferenceCoordinates, VirtualElement,
            mass::{ElementNodalLumpedMasses, LumpedMassVirtualElement},
            solid::ElementNodalStiffnessesSolid,
            time_scale::{fastest_time_scale, time_scale_exceeds},
        },
    },
};

/// The time scale of any union of the elements of a mesh as one virtual element.
pub struct Candidates<S = Vec<Vec<Vec<usize>>>> {
    boundary: Boundary<S>,
    elements_blocks: Vec<usize>,
    primitives: Vec<Primitive>,
    coordinates: NodalReferenceCoordinates,
    material: NeoHookean,
    stabilization: Scalar,
}

impl Candidates {
    /// The elements of a mesh, whose blocks can have different topologies.
    pub fn from_mesh(
        mesh: &Mesh<3>,
        poisson: Scalar,
        stabilization: Scalar,
    ) -> Result<Self, String> {
        Ok(Self::assemble(
            Boundary::try_from(mesh)?,
            mesh.elements_blocks(),
            Primitive::from_mesh(mesh),
            mesh.coordinates().clone(),
            poisson,
            stabilization,
        ))
    }
}

impl<S: ElementsFaces> Candidates<S> {
    /// Elements of one topology, which are in one block.
    pub fn new(
        elements_faces: S,
        coordinates: NodalReferenceCoordinates,
        poisson: Scalar,
        stabilization: Scalar,
    ) -> Self {
        let boundary = Boundary::new(elements_faces);
        let elements_blocks = vec![0; boundary.number_of_elements()];
        Self::assemble(
            boundary,
            elements_blocks,
            Vec::new(),
            coordinates,
            poisson,
            stabilization,
        )
    }
    fn assemble(
        boundary: Boundary<S>,
        elements_blocks: Vec<usize>,
        primitives: Vec<Primitive>,
        coordinates: NodalReferenceCoordinates,
        poisson: Scalar,
        stabilization: Scalar,
    ) -> Self {
        Self {
            boundary,
            elements_blocks,
            primitives,
            coordinates,
            material: NeoHookean {
                bulk_modulus: Stress::pascals(1.0 / (3.0 * (1.0 - 2.0 * poisson))),
                shear_modulus: Stress::pascals(0.5 / (1.0 + poisson)),
            },
            stabilization,
        }
    }
    pub fn number_of_elements(&self) -> usize {
        self.boundary.number_of_elements()
    }
    /// The fastest time scale of the union of the elements, zero if any part of it is inverted.
    pub fn time_scale(&self, elements: &[usize]) -> Result<Quantity<Time>, String> {
        let (element, coordinates) = self.element(elements)?;
        if inverted(&element) {
            return Ok(Time::seconds(0.0));
        }
        if let Some(time_scale) = self.finite_time_scale(elements) {
            return time_scale;
        }
        let (stiffnesses, masses) = self.matrices(&element, &coordinates)?;
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
        let (element, coordinates) = self.element(elements)?;
        if inverted(&element) {
            return Ok(false);
        }
        if let Some(exceeds) = self.finite_time_scale_exceeds(elements, minimum) {
            return exceeds;
        }
        let (stiffnesses, masses) = self.matrices(&element, &coordinates)?;
        Ok(time_scale_exceeds(&stiffnesses, &masses, minimum))
    }
    /// Whether the union of the elements can be one virtual element, else the reason it cannot.
    ///
    /// It must lie in one block, have a boundary that is one closed surface of genus zero, and
    /// be star-shaped about the mean of its nodes, with every one of the element's tetrahedra
    /// at least `minimum_volume` times the mean volume of them. The reason says which of those
    /// two it is, with how many tetrahedra have no volume or how small the smallest one is.
    pub fn check(&self, elements: &[usize], minimum_volume: Scalar) -> Result<(), String> {
        if elements
            .iter()
            .any(|&element| self.elements_blocks[element] != self.elements_blocks[elements[0]])
        {
            return Err("the elements are in different blocks".to_string());
        }
        self.boundary.surface(elements)?.ensure_sphere()?;
        let (element, _) = self.element(elements)?;
        let volumes = volumes(&element);
        let negative = volumes.iter().filter(|&&volume| volume <= 0.0).count();
        if negative > 0 {
            return Err(format!(
                "the element is not star-shaped about the mean of its nodes: {negative} of {} tetrahedra have no volume",
                volumes.len()
            ));
        }
        let mean = volumes.iter().sum::<Scalar>() / volumes.len() as Scalar;
        let smallest = volumes.iter().copied().fold(Scalar::INFINITY, Scalar::min) / mean;
        if smallest < minimum_volume {
            Err(format!(
                "a tetrahedron of the element is smaller than the minimum volume: {smallest:.3e} of the mean"
            ))
        } else {
            Ok(())
        }
    }
    fn matrices(
        &self,
        element: &Element,
        coordinates: &NodalReferenceCoordinates,
    ) -> Result<(ElementNodalStiffnessesSolid, ElementNodalLumpedMasses), String> {
        let stiffnesses = element
            .nodal_stiffnesses(&self.material, &coordinates.clone().into())
            .map_err(|error| format!("{error:?}"))?;
        let masses =
            element.nodal_lumped_masses(Density::kilograms_per_cubic_meter(1.0), coordinates);
        Ok((stiffnesses, masses))
    }
    fn element(&self, elements: &[usize]) -> Result<(Element, NodalReferenceCoordinates), String> {
        let faces = self.boundary.faces(elements)?;
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
        Ok((element, coordinates))
    }
}

fn inverted(element: &Element) -> bool {
    volumes(element).iter().any(|&volume| volume <= 0.0)
}

fn volumes(element: &Element) -> Vec<Scalar> {
    element
        .tetrahedra()
        .iter()
        .map(|tetrahedron| tetrahedron.volume().value())
        .collect()
}
