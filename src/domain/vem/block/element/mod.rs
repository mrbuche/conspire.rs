pub mod solid;

use crate::{
    defeat_message,
    math::{Scalar, Scalars, Tensor, TensorRank1Vec2D, TestError},
    mechanics::{CurrentCoordinatesRef, ReferenceCoordinate, Vectors, Vectors2D},
};
use std::{
    collections::VecDeque,
    fmt::{self, Debug, Display, Formatter},
};

pub type ElementNodalCoordinates<'a> = CurrentCoordinatesRef<'a>;
pub type ElementNodalReferenceCoordinates = TensorRank1Vec2D<3, 0>;
pub type GradientVectors = Vectors2D<0>;

pub struct Element {
    gradient_vectors: GradientVectors,
    integration_weights: Scalars,
}

pub trait VirtualElement
where
    Self: From<(ElementNodalReferenceCoordinates, Vec<usize>)>,
{
    fn gradient_vectors(&self) -> &GradientVectors;
    fn integration_weights(&self) -> &Scalars;
}

impl VirtualElement for Element {
    fn gradient_vectors(&self) -> &GradientVectors {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars {
        &self.integration_weights
    }
}

impl From<(ElementNodalReferenceCoordinates, Vec<usize>)> for Element {
    fn from(
        (reference_nodal_coordinates, element_nodes): (
            ElementNodalReferenceCoordinates,
            Vec<usize>,
        ),
    ) -> Self {
        let faces_info: Vec<(Vec<[ReferenceCoordinate; 3]>, ReferenceCoordinate, Scalar)> =
            reference_nodal_coordinates
                .into_iter()
                .map(|face_coordinates| {
                    let num_nodes_face = face_coordinates.len() as Scalar;
                    let face_center = face_coordinates
                        .iter()
                        .cloned()
                        .sum::<ReferenceCoordinate>()
                        / num_nodes_face;
                    let mut face_coordinates_one_ahead = VecDeque::from(face_coordinates.clone());
                    let first_entry = face_coordinates_one_ahead.pop_front().unwrap();
                    face_coordinates_one_ahead.push_back(first_entry);
                    (
                        face_coordinates
                            .into_iter()
                            .zip(face_coordinates_one_ahead)
                            .map(|(node_a_coordinates, node_b_coordinates)| {
                                let e_1 = &node_a_coordinates - &node_b_coordinates;
                                let e_2 = &node_b_coordinates - &face_center;
                                [node_a_coordinates, node_b_coordinates, e_1.cross(&e_2)]
                            })
                            .collect(),
                        face_center,
                        num_nodes_face,
                    )
                })
                .collect();
        let element_volume = faces_info
            .iter()
            .map(|(face_info, node_c, _)| {
                face_info
                    .iter()
                    .map(|[node_a, node_b, outward]| {
                        // (node_a[0] + node_b[0] + node_c[0]) * outward[0]
                        node_a * outward
                    })
                    .sum::<Scalar>()
            })
            .sum::<Scalar>()
            / 6.0;
        let integration_weights = Scalars::from([element_volume]);

        let num_nodes = element_nodes.len() as Scalar;
        // element_nodes.into_iter().map(|node|
        // NEED FACE NODES? THEN WOULD NEED ELEMENT FACES? THEN SHOULD JUST DO THE COORDINATES PART IN HERE INSTEAD OF IN BLOCK
        // ).collect()

        // let gradient_vectors = vec![
        //     (0..num_nodes).map(|node|
        //         faces_info
        //             .into_iter()
        //             .map(|(face_info, _, num_nodes_face)| {

        //             })
        //         .sum::<Vectors<0>>()
        //         * ((1.0 - 1.0 / num_nodes) / element_volume / 6.0),
        //     )
        // ]
        // .into();

        let gradient_vectors = vec![
            faces_info
                .into_iter()
                .map(|(face_info, _, num_nodes_face)| {
                    let mut face_info_one_back = VecDeque::from(face_info.clone());
                    let last_entry = face_info_one_back.pop_back().unwrap();
                    face_info_one_back.push_front(last_entry);
                    face_info
                        .into_iter()
                        .zip(face_info_one_back)
                        .map(|([_, _, outward], [_, _, outward_back])| outward + outward_back)
                        .collect::<Vectors<0>>()
                        * (1.0 + 1.0 / num_nodes_face)
                })
                .sum::<Vectors<0>>()
                * ((1.0 - 1.0 / num_nodes) / element_volume / 6.0),
        ]
        .into();
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

impl Debug for Element {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "VirtualElement {{ ... }}",)
    }
}

pub enum VirtualElementError {
    Upstream(String, String),
}

impl From<VirtualElementError> for TestError {
    fn from(error: VirtualElementError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl Debug for VirtualElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In virtual element: {element}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for VirtualElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In virtual element: {element}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

#[test]
fn temporary_poly() {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    use crate::vem::NodalReferenceCoordinates;
    let coordinates = NodalReferenceCoordinates::from(vec![
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, -phi, -1.0 / phi],
        [0.0, -phi, 1.0 / phi],
        [0.0, phi, -1.0 / phi],
        [0.0, phi, 1.0 / phi],
        [-phi, -1.0 / phi, 0.0],
        [-phi, 1.0 / phi, 0.0],
        [phi, -1.0 / phi, 0.0],
        [phi, 1.0 / phi, 0.0],
        [-1.0 / phi, 0.0, -phi],
        [1.0 / phi, 0.0, -phi],
        [-1.0 / phi, 0.0, phi],
        [1.0 / phi, 0.0, phi],
    ]);
    let face_node_connectivity = [
        [16, 17, 4, 8, 0],
        [12, 13, 2, 16, 0],
        [8, 9, 1, 12, 0],
        [9, 5, 19, 18, 1],
        [18, 3, 13, 12, 1],
        [10, 6, 17, 16, 2],
        [13, 3, 11, 10, 2],
        [7, 11, 3, 18, 19],
        [14, 5, 9, 8, 4],
        [6, 15, 14, 4, 17],
        [5, 14, 15, 7, 19],
        [6, 10, 11, 7, 15],
    ];
    let element_face_connectivity = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let element = Element::from((
        element_face_connectivity
            .iter()
            .map(|&face| {
                face_node_connectivity[face]
                    .iter()
                    .map(|&node| coordinates[node].clone())
                    .collect()
            })
            .collect::<ElementNodalReferenceCoordinates>(),
        face_node_connectivity.iter().flatten().copied().collect(),
    ));
    let length = (coordinates[face_node_connectivity[0][0]].clone()
        - coordinates[face_node_connectivity[0][1]].clone())
    .norm();
    let volume = (15.0 + 7.0 * 5.0_f64.sqrt()) / 4.0 * length.powi(3);
    assert!((element.integration_weights()[0] - volume).abs() < 1e-14);
    use crate::vem::NodalCoordinates;
    let coordinates_current = NodalCoordinates::from(coordinates);
    let coordinates_0 = coordinates_current
        .iter()
        .map(|coordinate| coordinate.into())
        .collect();
    use crate::vem::block::element::solid::SolidVirtualElement;
    element
        .deformation_gradients(coordinates_0)
        .iter()
        .for_each(|deformation_gradient| println!("{:?}", deformation_gradient))
}
