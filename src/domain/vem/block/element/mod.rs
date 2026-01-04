pub mod solid;

use crate::{
    defeat_message,
    math::{Scalar, Scalars, Tensor, TensorRank1Vec2D, TestError},
    mechanics::{CurrentCoordinatesRef, ReferenceCoordinate, Vectors2D},
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
    for<'a> Self: From<(
        ElementNodalReferenceCoordinates,
        Vec<usize>,
        &'a [usize],
        &'a [Vec<usize>],
    )>,
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

impl
    From<(
        ElementNodalReferenceCoordinates,
        Vec<usize>,
        &[usize],
        &[Vec<usize>],
    )> for Element
{
    fn from(
        (reference_nodal_coordinates, element_faces, element_nodes, faces_nodes): (
            ElementNodalReferenceCoordinates,
            Vec<usize>,
            &[usize],
            &[Vec<usize>],
        ),
    ) -> Self {
        let element_volume = reference_nodal_coordinates
            .iter()
            .map(|face_coordinates| {
                let face_center = face_coordinates
                    .iter()
                    .cloned()
                    .sum::<ReferenceCoordinate>()
                    / (face_coordinates.len() as Scalar);
                let mut face_coordinates_one_ahead = VecDeque::from(face_coordinates.clone());
                let first_entry = face_coordinates_one_ahead.pop_front().unwrap();
                face_coordinates_one_ahead.push_back(first_entry);
                face_coordinates
                    .into_iter()
                    .zip(face_coordinates_one_ahead)
                    .map(|(node_a_coordinates, node_b_coordinates)| {
                        let e_1 = node_a_coordinates - &node_b_coordinates;
                        let e_2 = &node_b_coordinates - &face_center;
                        node_a_coordinates * e_1.cross(&e_2)
                    })
                    .sum::<Scalar>()
            })
            .sum::<Scalar>()
            / 6.0;
        let integration_weights = Scalars::from([element_volume]);
        let gradient_vectors = vec![
            element_nodes
                .iter()
                .map(|&node| {
                    element_faces
                        .iter()
                        .zip(reference_nodal_coordinates.iter())
                        .filter_map(|(&face, face_coordinates)| {
                            if faces_nodes[face].contains(&node) {
                                let num_nodes_face = face_coordinates.len() as Scalar;
                                let face_center = face_coordinates
                                    .iter()
                                    .cloned()
                                    .sum::<ReferenceCoordinate>()
                                    / num_nodes_face;
                                let mut face_coordinates_one_ahead =
                                    VecDeque::from(face_coordinates.clone());
                                let first_entry = face_coordinates_one_ahead.pop_front().unwrap();
                                face_coordinates_one_ahead.push_back(first_entry);
                                Some(
                                    face_coordinates
                                        .into_iter()
                                        .zip(face_coordinates_one_ahead)
                                        .zip(faces_nodes[face].iter())
                                        .map(
                                            |(
                                                (node_a_coordinates, node_b_coordinates),
                                                &node_a,
                                            )| {
                                                let node_a_spot = faces_nodes[face]
                                                    .iter()
                                                    .position(|&n| n == node_a)
                                                    .unwrap();
                                                let node_b =
                                                    if node_a_spot + 1 == faces_nodes[face].len() {
                                                        faces_nodes[face][0]
                                                    } else {
                                                        faces_nodes[face][node_a_spot + 1]
                                                    };
                                                let factor = if node == node_a || node == node_b {
                                                    1.0 + 1.0 / num_nodes_face
                                                } else {
                                                    1.0 / num_nodes_face
                                                };
                                                let e_1 = &node_b_coordinates - node_a_coordinates;
                                                let e_2 = &face_center - node_b_coordinates;
                                                e_1.cross(&e_2) * factor
                                            },
                                        )
                                        .sum::<ReferenceCoordinate>(),
                                )
                            } else {
                                None
                            }
                        })
                        .sum::<ReferenceCoordinate>()
                        / (element_volume * 6.0)
                })
                .collect(),
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
fn temporary_poly_0() {
    use crate::vem::NodalReferenceCoordinates;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
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
    let face_node_connectivity = vec![
        vec![16, 17, 4, 8, 0],
        vec![12, 13, 2, 16, 0],
        vec![8, 9, 1, 12, 0],
        vec![9, 5, 19, 18, 1],
        vec![18, 3, 13, 12, 1],
        vec![10, 6, 17, 16, 2],
        vec![13, 3, 11, 10, 2],
        vec![7, 11, 3, 18, 19],
        vec![14, 5, 9, 8, 4],
        vec![6, 15, 14, 4, 17],
        vec![5, 14, 15, 7, 19],
        vec![6, 10, 11, 7, 15],
    ];
    let element_face_connectivity = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]];
    use crate::constitutive::solid::hyperelastic::NeoHookean;
    use crate::vem::block::{
        Block,
        solid::{SolidVirtualElementBlock, elastic::ElasticVirtualElementBlock},
    };
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: 3.0,
            bulk_modulus: 13.0,
        },
        coordinates.clone(),
        element_face_connectivity.clone(),
        face_node_connectivity.clone(),
    ));
    use crate::domain::NodalForcesSolid;
    use crate::math::{TensorArray, assert_eq_within_tols};
    use crate::mechanics::DeformationGradient;
    use crate::vem::NodalCoordinates;
    let coordinates_current = NodalCoordinates::from(coordinates.clone());
    assert_eq_within_tols(
        &DeformationGradient::identity(),
        &block.deformation_gradients(&coordinates_current)[0][0],
    )
    .unwrap();
    assert_eq_within_tols(
        &NodalForcesSolid::zero(coordinates_current.len()),
        &block.nodal_forces(&coordinates_current).unwrap(),
    )
    .unwrap();
    let length = (coordinates[face_node_connectivity[0][0]].clone()
        - coordinates[face_node_connectivity[0][1]].clone())
    .norm();
    let volume = (15.0 + 7.0 * 5.0_f64.sqrt()) / 4.0 * length.powi(3);
    assert!((block.elements()[0].integration_weights()[0] - volume).abs() < 1e-14);
}

#[test]
fn temporary_poly_1() {
    use crate::vem::NodalReferenceCoordinates;
    let coordinates = NodalReferenceCoordinates::from(vec![
        [-0.7727027, -0.65398245, -0.80050964],
        [-0.55585269, -1.31907453, 1.32652506],
        [-0.68068751, 0.86362469, -0.58348725],
        [-1.2475506, 1.06566759, 1.45034587],
        [1.47277602, -1.10640079, -0.90724596],
        [1.10274756, -0.69153902, 1.27617253],
        [0.64323505, 1.36639746, -1.48447683],
        [0.91277928, 0.97322043, 0.67055],
        [-0.19978796, -2.0201241, -0.50145446],
        [-0.07547771, -1.54630032, 0.22127876],
        [0.37534904, 1.50203587, -0.81372091],
        [-0.20273152, 1.4672534, 0.27738481],
        [-1.98854772, -0.25595864, 0.16143842],
        [-1.80085125, 0.19913772, -0.19452172],
        [1.3154974, -0.72436122, 0.17437191],
        [2.09624968, 1.01585944, 0.29687302],
        [-0.61664715, 0.18078644, -1.94806432],
        [0.86740811, -0.38259605, -1.2754194],
        [-1.08169702, -0.39837623, 1.63255916],
        [0.12293689, -0.48172557, 1.4158596],
    ]);
    let face_node_connectivity = vec![
        vec![16, 17, 4, 8, 0],
        vec![12, 13, 2, 16, 0],
        vec![8, 9, 1, 12, 0],
        vec![9, 5, 19, 18, 1],
        vec![18, 3, 13, 12, 1],
        vec![10, 6, 17, 16, 2],
        vec![13, 3, 11, 10, 2],
        vec![7, 11, 3, 18, 19],
        vec![14, 5, 9, 8, 4],
        vec![6, 15, 14, 4, 17],
        vec![5, 14, 15, 7, 19],
        vec![6, 10, 11, 7, 15],
    ];
    let element_face_connectivity = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]];
    use crate::constitutive::solid::hyperelastic::NeoHookean;
    use crate::vem::block::{
        Block,
        solid::{SolidVirtualElementBlock, elastic::ElasticVirtualElementBlock},
    };
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: 3.0,
            bulk_modulus: 13.0,
        },
        coordinates.clone(),
        element_face_connectivity.clone(),
        face_node_connectivity.clone(),
    ));
    use crate::domain::NodalForcesSolid;
    use crate::math::{TensorArray, assert_eq_within_tols};
    use crate::mechanics::DeformationGradient;
    use crate::vem::NodalCoordinates;
    let coordinates_current = NodalCoordinates::from(coordinates.clone());
    assert_eq_within_tols(
        &DeformationGradient::identity(),
        &block.deformation_gradients(&coordinates_current)[0][0],
    )
    .unwrap();
    assert_eq_within_tols(
        &NodalForcesSolid::zero(coordinates_current.len()),
        &block.nodal_forces(&coordinates_current).unwrap(),
    )
    .unwrap();
    use crate::mechanics::test::{get_deformation_gradient, get_translation_current_configuration};
    let coordinates_current: NodalCoordinates = coordinates
        .iter()
        .map(|coord| get_deformation_gradient() * coord + get_translation_current_configuration())
        .collect();
    assert_eq_within_tols(
        &get_deformation_gradient(),
        &block.deformation_gradients(&coordinates_current)[0][0],
    )
    .unwrap();
}
