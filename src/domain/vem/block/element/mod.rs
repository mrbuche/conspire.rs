use crate::math::Reference;
pub mod solid;
#[cfg(test)]
mod test;

use crate::{
    domain::block::element::{ElementError, ElementKind},
    fem::block::element::{
        ElementNodalReferenceCoordinates as FemElementNodalReferenceCoordinates, FiniteElement,
        linear::Tetrahedron,
    },
    math::{
        CrossProduct, Current, Quantity, Scalar, Tensor, TensorArray, TensorRank1, TensorRank1List,
        TensorRank1Vec, TensorRank1Vec2D, TensorVector,
    },
    mechanics::ReferenceCoordinate,
    units::{Area, Length, ReciprocalLength, Velocity, Volume},
    vem::{NodalCoordinates, NodalReferenceCoordinates, NodalVelocities},
};

use std::fmt::{self, Debug, Display, Formatter};

pub type ElementNodalCoordinates = NodalCoordinates;
pub type ElementNodalVelocities = NodalVelocities;
pub type ElementNodalReferenceCoordinates = TensorRank1Vec2D<3, Reference, Length>;
pub type GradientVectors = TensorRank1Vec2D<3, Reference, ReciprocalLength>;
pub type IntegrationWeights = TensorVector<Quantity<Volume>>;

pub type TetrahedraQuantities<U> = Vec<TensorRank1List<3, Current, 4, U>>;
pub type TetrahedraCoordinates = TetrahedraQuantities<Length>;
pub type TetrahedraVelocities = TetrahedraQuantities<Velocity>;

pub struct Element {
    faces_nodes: Vec<Vec<usize>>,
    gradient_vectors: GradientVectors,
    integration_weights: IntegrationWeights,
    stabilization: Scalar,
    tetrahedra: Vec<Tetrahedron>,
    tetrahedra_nodes: Vec<[usize; 3]>,
}

impl Element {
    pub(crate) fn upstream(&self, error: impl Display) -> VirtualElementError {
        VirtualElementError::upstream(error, self)
    }
}

pub trait VirtualElement
where
    for<'a> Self: From<(
        ElementNodalReferenceCoordinates,
        &'a [usize],
        &'a [usize],
        &'a [Vec<usize>],
    )>,
{
    fn element_center<U>(
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TensorRank1<3, Current, U>;
    fn faces_centers<U>(
        &self,
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TensorRank1Vec<3, Current, U>;
    fn faces_nodes(&self) -> &[Vec<usize>];
    fn gradient_vectors(&self) -> &GradientVectors;
    fn integration_weights(&self) -> &IntegrationWeights;
    fn stabilization(&self) -> Scalar;
    fn tetrahedra(&self) -> &[Tetrahedron];
    fn tetrahedra_coordinates<U>(
        &self,
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TetrahedraQuantities<U>;
    fn tetrahedra_nodes(&self) -> &[[usize; 3]];
}

impl VirtualElement for Element {
    fn element_center<U>(
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TensorRank1<3, Current, U> {
        nodal_quantities
            .iter()
            .cloned()
            .sum::<TensorRank1<3, Current, U>>()
            / nodal_quantities.len() as Scalar
    }
    fn faces_centers<U>(
        &self,
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TensorRank1Vec<3, Current, U> {
        self.faces_nodes()
            .iter()
            .map(|face_nodes| {
                face_nodes
                    .iter()
                    .map(|&face_node| nodal_quantities[face_node].clone())
                    .sum::<TensorRank1<3, Current, U>>()
                    / (face_nodes.len() as Scalar)
            })
            .collect()
    }
    fn faces_nodes(&self) -> &[Vec<usize>] {
        &self.faces_nodes
    }
    fn gradient_vectors(&self) -> &GradientVectors {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &IntegrationWeights {
        &self.integration_weights
    }
    fn stabilization(&self) -> Scalar {
        self.stabilization
    }
    fn tetrahedra(&self) -> &[Tetrahedron] {
        &self.tetrahedra
    }
    fn tetrahedra_coordinates<U>(
        &self,
        nodal_quantities: &TensorRank1Vec<3, Current, U>,
    ) -> TetrahedraQuantities<U> {
        let element_center = Self::element_center(nodal_quantities);
        let faces_centers = self.faces_centers(nodal_quantities);
        self.tetrahedra_nodes()
            .iter()
            .map(|&[face, node_b, node_a]| {
                [
                    faces_centers[face].clone(),
                    nodal_quantities[node_b].clone(),
                    nodal_quantities[node_a].clone(),
                    element_center.clone(),
                ]
                .into()
            })
            .collect()
    }
    fn tetrahedra_nodes(&self) -> &[[usize; 3]] {
        &self.tetrahedra_nodes
    }
}

impl
    From<(
        ElementNodalReferenceCoordinates,
        &[usize],
        &[usize],
        &[Vec<usize>],
    )> for Element
{
    fn from(
        (reference_nodal_coordinates, element_faces, element_nodes, block_faces_nodes): (
            ElementNodalReferenceCoordinates,
            &[usize],
            &[usize],
            &[Vec<usize>],
        ),
    ) -> Self {
        let faces_nodes = element_faces
            .iter()
            .map(|&element_face| {
                block_faces_nodes[element_face]
                    .iter()
                    .map(|face_node| {
                        element_nodes
                            .iter()
                            .position(|element_node| face_node == element_node)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut nodal_coordinates =
            NodalReferenceCoordinates::from(vec![
                ReferenceCoordinate::from([0.0, 0.0, 0.0]);
                element_nodes.len()
            ]);
        faces_nodes
            .iter()
            .zip(reference_nodal_coordinates.iter())
            .for_each(|(face_nodes, face_coordinates)| {
                face_nodes
                    .iter()
                    .zip(face_coordinates.iter())
                    .for_each(|(&node, coordinates)| nodal_coordinates[node] = coordinates.clone())
            });
        let element_center = nodal_coordinates.into_iter().sum::<ReferenceCoordinate>()
            / (element_nodes.len() as Scalar);
        let mut area_vectors = vec![TensorRank1::<3, Reference, Area>::zero(); element_nodes.len()];
        let tetrahedra_nodes = faces_nodes
            .iter()
            .enumerate()
            .flat_map(|(face, face_nodes)| {
                (0..face_nodes.len())
                    .map(|spot| {
                        [
                            face,
                            face_nodes[(spot + 1) % face_nodes.len()],
                            face_nodes[spot],
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let tetrahedra = faces_nodes
            .iter()
            .zip(reference_nodal_coordinates.iter())
            .flat_map(|(face_nodes, face_coordinates)| {
                let num_nodes_face = face_coordinates.len();
                let face_center = face_coordinates
                    .iter()
                    .cloned()
                    .sum::<ReferenceCoordinate>()
                    / (num_nodes_face as Scalar);
                let mut face_area_vector = TensorRank1::<3, Reference, Area>::zero();
                let face_tetrahedra = (0..num_nodes_face)
                    .map(|spot| {
                        let next = (spot + 1) % num_nodes_face;
                        let e_1 = &face_coordinates[next] - &face_coordinates[spot];
                        let e_2 = &face_center - &face_coordinates[next];
                        let cross = e_1.cross(&e_2);
                        face_area_vector += &cross;
                        area_vectors[face_nodes[spot]] += &cross;
                        area_vectors[face_nodes[next]] += &cross;
                        Tetrahedron::from(FemElementNodalReferenceCoordinates::from([
                            face_center.clone(),
                            face_coordinates[next].clone(),
                            face_coordinates[spot].clone(),
                            element_center.clone(),
                        ]))
                    })
                    .collect::<Vec<_>>();
                let shared = &face_area_vector / (num_nodes_face as Scalar);
                face_nodes
                    .iter()
                    .for_each(|&node| area_vectors[node] += &shared);
                face_tetrahedra
            })
            .collect::<Vec<_>>();
        let element_volume = tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume())
            .sum::<Quantity<Volume>>();
        let gradient_vectors = GradientVectors::from(vec![
            area_vectors
                .into_iter()
                .map(|area_vector| area_vector / (element_volume * 6.0))
                .collect::<TensorRank1Vec<3, Reference, ReciprocalLength>>(),
        ]);
        let integration_weights = IntegrationWeights::from([element_volume]);
        Self {
            faces_nodes,
            gradient_vectors,
            integration_weights,
            stabilization: 0.1,
            tetrahedra,
            tetrahedra_nodes,
        }
    }
}

impl Debug for Element {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "VirtualElement {{ ... }}",)
    }
}

pub struct VirtualElementKind;

impl ElementKind for VirtualElementKind {
    const NAME: &'static str = "virtual element";
}

pub type VirtualElementError = ElementError<VirtualElementKind>;
