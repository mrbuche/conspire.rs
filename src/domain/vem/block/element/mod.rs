use crate::math::Reference;
pub mod solid;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates as FemElementNodalCoordinates,
        ElementNodalReferenceCoordinates as FemElementNodalReferenceCoordinates, FiniteElement,
        linear::Tetrahedron,
    },
    math::{
        CrossProduct, Quantity, Scalar, Style, StyledError, Tensor, TensorArray, TensorRank1,
        TensorRank1Vec, TensorRank1Vec2D, TensorVector, assert::AssertionError, styled_error,
    },
    mechanics::{CurrentCoordinate, CurrentCoordinatesRef, ReferenceCoordinate},
    units::{Area, Length, ReciprocalLength, Volume},
    vem::{NodalCoordinates, NodalReferenceCoordinates},
};

#[cfg(test)]
use crate::math::assert::Assert;
use std::fmt::{self, Debug, Display, Formatter};

pub type ElementNodalCoordinates<'a> = CurrentCoordinatesRef<'a>;
pub type ElementNodalReferenceCoordinates = TensorRank1Vec2D<3, Reference, Length>;
pub type GradientVectors = TensorRank1Vec2D<3, Reference, ReciprocalLength>;
pub type IntegrationWeights = TensorVector<Quantity<Volume>>;

pub type TetrahedraCoordinates = Vec<FemElementNodalCoordinates<4>>;

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
        VirtualElementError::Upstream(format!("{error}"), format!("{self:?}"))
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
    fn element_center<'a>(nodal_coordinates: &ElementNodalCoordinates<'a>) -> CurrentCoordinate;
    fn faces_centers<'a>(
        &'a self,
        nodal_coordinates: &ElementNodalCoordinates<'a>,
    ) -> NodalCoordinates;
    fn faces_nodes(&self) -> &[Vec<usize>];
    fn gradient_vectors(&self) -> &GradientVectors;
    fn integration_weights(&self) -> &IntegrationWeights;
    fn stabilization(&self) -> Scalar;
    fn tetrahedra(&self) -> &[Tetrahedron];
    fn tetrahedra_coordinates<'a>(
        &'a self,
        nodal_coordinates: &ElementNodalCoordinates<'a>,
    ) -> TetrahedraCoordinates;
    fn tetrahedra_nodes(&self) -> &[[usize; 3]];
}

impl VirtualElement for Element {
    fn element_center<'a>(nodal_coordinates: &ElementNodalCoordinates<'a>) -> CurrentCoordinate {
        nodal_coordinates
            .iter()
            .map(|&nodal_coordinate| nodal_coordinate.clone())
            .sum::<CurrentCoordinate>()
            / nodal_coordinates.len() as Scalar
    }
    fn faces_centers<'a>(
        &'a self,
        nodal_coordinates: &ElementNodalCoordinates<'a>,
    ) -> NodalCoordinates {
        self.faces_nodes()
            .iter()
            .map(|face_nodes| {
                face_nodes
                    .iter()
                    .map(|&face_node| nodal_coordinates[face_node].clone())
                    .sum::<CurrentCoordinate>()
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
    fn tetrahedra_coordinates<'a>(
        &'a self,
        nodal_coordinates: &ElementNodalCoordinates<'a>,
    ) -> TetrahedraCoordinates {
        let element_center = Self::element_center(nodal_coordinates);
        let faces_centers = self.faces_centers(nodal_coordinates);
        self.tetrahedra_nodes()
            .iter()
            .map(|&[face, node_b, node_a]| {
                [
                    faces_centers[face].clone(),
                    nodal_coordinates[node_b].clone(),
                    nodal_coordinates[node_a].clone(),
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
                (0..num_nodes_face)
                    .map(|spot| {
                        let next = (spot + 1) % num_nodes_face;
                        let e_1 = &face_coordinates[next] - &face_coordinates[spot];
                        let e_2 = &face_center - &face_coordinates[next];
                        let cross = e_1.cross(&e_2);
                        let shared = &cross / (num_nodes_face as Scalar);
                        face_nodes
                            .iter()
                            .for_each(|&node| area_vectors[node] += &shared);
                        area_vectors[face_nodes[spot]] += &cross;
                        area_vectors[face_nodes[next]] += &cross;
                        Tetrahedron::from(FemElementNodalReferenceCoordinates::from([
                            face_center.clone(),
                            face_coordinates[next].clone(),
                            face_coordinates[spot].clone(),
                            element_center.clone(),
                        ]))
                    })
                    .collect::<Vec<_>>()
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

pub enum VirtualElementError {
    Upstream(String, String),
}

impl VirtualElementError {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
}

impl From<VirtualElementError> for AssertionError {
    fn from(error: VirtualElementError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl StyledError for VirtualElementError {
    fn message(&self, style: &Style) -> String {
        let c = style.frame;
        match self {
            Self::Upstream(error, element) => format!(
                "{error}{c}\n\
                In virtual element: {element}."
            ),
        }
    }
}

styled_error!(VirtualElementError);

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
    use crate::fem::solid::elastic::ElasticElements;
    use crate::units::Stress;
    use crate::vem::block::{Block, solid::SolidVirtualElements};
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        element_face_connectivity.clone(),
        face_node_connectivity.clone(),
        &coordinates,
    ));
    use crate::fem::solid::NodalForcesSolid;
    use crate::math::TensorArray;
    use crate::mechanics::DeformationGradient;
    use crate::vem::NodalCoordinates;
    let coordinates_current = NodalCoordinates::from(coordinates.clone());
    Assert::default()
        .eq_within_tols(
            DeformationGradient::identity(),
            &block.deformation_gradients(&coordinates_current)[0][0],
        )
        .unwrap();
    Assert::default()
        .eq_within_tols(
            NodalForcesSolid::zero(coordinates_current.len()),
            &block.nodal_forces(&coordinates_current).unwrap(),
        )
        .unwrap();
    let length = (coordinates[face_node_connectivity[0][0]].clone()
        - coordinates[face_node_connectivity[0][1]].clone())
    .norm();
    let volume = length * length * length * ((15.0 + 7.0 * 5.0_f64.sqrt()) / 4.0);
    assert!((block.elements()[0].integration_weights()[0] / volume - 1.0).abs() < 1e-14);
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
    use crate::fem::solid::elastic::ElasticElements;
    use crate::units::Stress;
    use crate::vem::block::{Block, solid::SolidVirtualElements};
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        element_face_connectivity.clone(),
        face_node_connectivity.clone(),
        &coordinates,
    ));
    use crate::fem::solid::NodalForcesSolid;
    use crate::math::TensorArray;
    use crate::mechanics::DeformationGradient;
    use crate::vem::NodalCoordinates;
    let coordinates_current = NodalCoordinates::from(coordinates.clone());
    Assert::default()
        .eq_within_tols(
            DeformationGradient::identity(),
            &block.deformation_gradients(&coordinates_current)[0][0],
        )
        .unwrap();
    Assert::default()
        .eq_within_tols(
            NodalForcesSolid::zero(coordinates_current.len()),
            &block.nodal_forces(&coordinates_current).unwrap(),
        )
        .unwrap();
    use crate::mechanics::test::{get_deformation_gradient, get_translation_current_configuration};
    let coordinates_current: NodalCoordinates = coordinates
        .iter()
        .map(|coord| get_deformation_gradient() * coord + get_translation_current_configuration())
        .collect();
    Assert::default()
        .eq_within_tols(
            get_deformation_gradient(),
            &block.deformation_gradients(&coordinates_current)[0][0],
        )
        .unwrap();
}

#[test]
fn temporary_poly_2() {
    use crate::vem::NodalReferenceCoordinates;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let coordinates_0 = NodalReferenceCoordinates::from(vec![
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
    use crate::fem::solid::elastic::ElasticElements;
    use crate::units::Stress;
    use crate::vem::block::Block;
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        element_face_connectivity.clone(),
        face_node_connectivity.clone(),
        &coordinates_0,
    ));
    use crate::vem::NodalCoordinates;
    let coordinates = NodalCoordinates::from(vec![
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
    use crate::EPSILON;
    use crate::fem::solid::hyperelastic::HyperelasticElements;
    let mut finite_difference = crate::math::Quantity::default();
    let nodal_forces_fd = (0..coordinates.len())
        .map(|node| {
            (0..3)
                .map(|i| {
                    let mut nodal_coordinates = coordinates.clone();
                    nodal_coordinates[node][i] += crate::math::assert::perturbation(0.5 * EPSILON);
                    finite_difference = block.helmholtz_free_energy(&nodal_coordinates).unwrap();
                    nodal_coordinates[node][i] -= crate::math::assert::perturbation(EPSILON);
                    finite_difference -= block.helmholtz_free_energy(&nodal_coordinates).unwrap();
                    finite_difference
                        / crate::math::assert::perturbation::<crate::units::Length>(EPSILON)
                })
                .collect()
        })
        .collect();
    Assert::default()
        .eq_within_fd_tol(block.nodal_forces(&coordinates).unwrap(), &nodal_forces_fd)
        .unwrap();
    let mut finite_difference = crate::math::Quantity::default();
    let nodal_stiffnesses_fd = (0..coordinates.len())
        .map(|a| {
            (0..coordinates.len())
                .map(|b| {
                    (0..3)
                        .map(|i| {
                            (0..3)
                                .map(|j| {
                                    let mut nodal_coordinates = coordinates.clone();
                                    nodal_coordinates[b][j] +=
                                        crate::math::assert::perturbation(0.5 * EPSILON);
                                    finite_difference =
                                        block.nodal_forces(&nodal_coordinates).unwrap()[a][i];
                                    nodal_coordinates[b][j] -=
                                        crate::math::assert::perturbation(EPSILON);
                                    finite_difference -=
                                        block.nodal_forces(&nodal_coordinates).unwrap()[a][i];
                                    finite_difference
                                        / crate::math::assert::perturbation::<crate::units::Length>(
                                            EPSILON,
                                        )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    Assert::default()
        .eq_within_fd_tol(
            block.nodal_stiffnesses(&coordinates).unwrap(),
            &nodal_stiffnesses_fd,
        )
        .unwrap();
}

#[test]
fn temporary_poly_3() {
    use crate::{
        constitutive::solid::hyperelastic::NeoHookean,
        fem::solid::elastic::ElasticElements,
        units::Stress,
        vem::{
            NodalCoordinates, NodalReferenceCoordinates,
            block::{Block, solid::SolidVirtualElements},
        },
    };
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let unit = [
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
    ];
    let unit_faces = [
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
    // element 0 owns faces 12..24 and element 1 owns faces 0..12, so that the
    // element ordering does not match the face ordering
    let coordinates = NodalReferenceCoordinates::from(
        (0..2)
            .flat_map(|copy| {
                unit.iter()
                    .map(|coordinate| {
                        [
                            coordinate[0] + 4.0 * copy as Scalar,
                            coordinate[1],
                            coordinate[2],
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    );
    let faces_nodes = (0..2)
        .flat_map(|copy| {
            unit_faces
                .iter()
                .map(|face| {
                    face.iter()
                        .map(|&node| node + (1 - copy) * unit.len())
                        .collect()
                })
                .collect::<Vec<Vec<_>>>()
        })
        .collect();
    let block = Block::<_, Element>::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        vec![(12..24).collect::<Vec<_>>(), (0..12).collect::<Vec<_>>()],
        faces_nodes,
        &coordinates,
    ));
    use crate::{fem::solid::NodalForcesSolid, math::TensorArray, mechanics::DeformationGradient};
    let coordinates = NodalCoordinates::from(coordinates);
    block
        .deformation_gradients(&coordinates)
        .iter()
        .for_each(|deformation_gradients| {
            Assert::default()
                .eq_within_tols(DeformationGradient::identity(), &deformation_gradients[0])
                .unwrap()
        });
    Assert::default()
        .eq_within_tols(
            NodalForcesSolid::zero(coordinates.len()),
            &block.nodal_forces(&coordinates).unwrap(),
        )
        .unwrap();
}
