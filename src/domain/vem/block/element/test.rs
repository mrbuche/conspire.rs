use crate::{
    domain::block::element::solid::SolidElement,
    math::{Rank2, Tensor, TensorArray, assert::AssertionError},
    mechanics::{
        DeformationGradient, DeformationGradientRate, DeformationGradientRates,
        DeformationGradients,
        test::{
            get_deformation_gradient, get_deformation_gradient_rate,
            get_rotation_current_configuration, get_rotation_rate_current_configuration,
            get_rotation_reference_configuration, get_translation_current_configuration,
            get_translation_rate_current_configuration, get_translation_reference_configuration,
        },
    },
    vem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::element::{Element, ElementNodalReferenceCoordinates, VirtualElement},
    },
};

type DeformationGradientList = DeformationGradients;
type DeformationGradientRateList = DeformationGradientRates;

fn element_faces() -> Vec<usize> {
    (0..12).collect()
}

fn element_nodes() -> Vec<usize> {
    (0..20).collect()
}

fn faces_nodes() -> Vec<Vec<usize>> {
    vec![
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
    ]
}

fn reference_coordinates() -> NodalReferenceCoordinates {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    NodalReferenceCoordinates::from(vec![
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
    ])
}

fn reference_coordinates_transformed() -> NodalReferenceCoordinates {
    reference_coordinates()
        .iter()
        .map(|reference_coordinate| {
            get_rotation_reference_configuration() * reference_coordinate
                + get_translation_reference_configuration()
        })
        .collect()
}

fn coordinates() -> NodalCoordinates {
    reference_coordinates()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient() * reference_coordinate)
        .collect()
}

fn coordinates_transformed() -> NodalCoordinates {
    coordinates()
        .iter()
        .map(|coordinate| {
            get_rotation_current_configuration() * coordinate
                + get_translation_current_configuration()
        })
        .collect()
}

fn velocities() -> NodalVelocities {
    reference_coordinates()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient_rate() * reference_coordinate)
        .collect()
}

fn velocities_transformed() -> NodalVelocities {
    coordinates()
        .iter()
        .zip(velocities().iter())
        .map(|(coordinate, velocity)| {
            get_rotation_current_configuration() * velocity
                + get_rotation_rate_current_configuration() * coordinate
                + get_translation_rate_current_configuration()
        })
        .collect()
}

fn element_nodal_reference_coordinates(
    reference: &NodalReferenceCoordinates,
) -> ElementNodalReferenceCoordinates {
    element_faces()
        .iter()
        .map(|&face| {
            faces_nodes()[face]
                .iter()
                .map(|&node| reference[node].clone())
                .collect()
        })
        .collect()
}

fn element() -> Element {
    Element::from((
        element_nodal_reference_coordinates(&reference_coordinates()),
        element_faces().as_slice(),
        element_nodes().as_slice(),
        faces_nodes().as_slice(),
    ))
}

fn element_transformed() -> Element {
    Element::from((
        element_nodal_reference_coordinates(&reference_coordinates_transformed()),
        element_faces().as_slice(),
        element_nodes().as_slice(),
        faces_nodes().as_slice(),
    ))
}

fn number_of_gradients() -> usize {
    element().gradient_vectors().len()
}

fn identity_deformation_gradients() -> DeformationGradientList {
    (0..number_of_gradients())
        .map(|_| DeformationGradient::identity())
        .collect()
}

fn zero_deformation_gradient_rates() -> DeformationGradientRateList {
    (0..number_of_gradients())
        .map(|_| DeformationGradientRate::zero())
        .collect()
}

fn zero_velocities() -> NodalVelocities {
    NodalVelocities::zero(reference_coordinates().len())
}

crate::domain::block::element::test::test_solid_deformation_gradient!();
