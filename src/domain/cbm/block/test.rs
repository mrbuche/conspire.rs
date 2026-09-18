use super::Block;
use crate::{
    EPSILON,
    cbm::SolidElements,
    constitutive::solid::elastic::AlmansiHamelEulerian,
    domain::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities, nodal_coordinates,
        solid::{NodalStiffnessesSolid, elastic::ElasticElements},
    },
    geometry::{Coordinate, Coordinates, mesh::PrimitiveConnectivity},
    math::{Tensor, assert::Assert, assert::perturbation},
    mechanics::{DeformationGradient, DeformationGradientRate},
    units::{Length, Stress},
};

fn two_tetrahedra_reference() -> (PrimitiveConnectivity<3, 4>, NodalReferenceCoordinates<3>) {
    let connectivity = PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 4, 2, 3]]);
    let coordinates: Coordinates<3> = vec![
        Coordinate::from([0.0, 0.0, 0.0]),
        Coordinate::from([1.0, 0.0, 0.0]),
        Coordinate::from([0.0, 1.0, 0.0]),
        Coordinate::from([0.0, 0.0, 1.0]),
        Coordinate::from([1.0, 1.0, 1.0]),
    ]
    .into_iter()
    .collect();
    (connectivity, nodal_coordinates(coordinates))
}

fn apply(
    deformation_gradient: &DeformationGradient,
    reference: &NodalReferenceCoordinates<3>,
) -> NodalCoordinates<3> {
    reference
        .iter()
        .map(|reference_coordinate| deformation_gradient * reference_coordinate)
        .collect()
}

#[test]
fn patch_test_uniform_deformation_gradient() {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let block = Block::from(((), connectivity, &reference_coordinates));
    let deformation_gradient =
        DeformationGradient::from([[1.1, 0.05, 0.0], [0.0, 0.9, 0.02], [-0.03, 0.0, 1.2]]);
    let current_coordinates = apply(&deformation_gradient, &reference_coordinates);
    block
        .deformation_gradients(&current_coordinates)
        .iter()
        .try_for_each(|particle_deformation_gradient| {
            Assert::default().eq_within_tols(particle_deformation_gradient, &deformation_gradient)
        })
        .unwrap()
}

#[test]
fn patch_test_uniform_deformation_gradient_rate() {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let block = Block::from(((), connectivity, &reference_coordinates));
    let deformation_gradient =
        DeformationGradient::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let current_coordinates = apply(&deformation_gradient, &reference_coordinates);
    let deformation_gradient_rate =
        DeformationGradientRate::from([[0.1, 0.0, 0.0], [0.0, -0.05, 0.01], [0.0, 0.0, 0.02]]);
    let velocities: NodalVelocities<3> = reference_coordinates
        .iter()
        .map(|reference_coordinate| &deformation_gradient_rate * reference_coordinate)
        .collect();
    block
        .deformation_gradient_rates(&current_coordinates, &velocities)
        .iter()
        .try_for_each(|particle_deformation_gradient_rate| {
            Assert::default().eq_within_tols(
                particle_deformation_gradient_rate,
                &deformation_gradient_rate,
            )
        })
        .unwrap()
}

fn constitutive_model() -> AlmansiHamelEulerian {
    AlmansiHamelEulerian {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn non_uniformly_deformed_coordinates(
    reference_coordinates: &NodalReferenceCoordinates<3>,
) -> NodalCoordinates<3> {
    let deformation_gradient =
        DeformationGradient::from([[1.05, 0.02, 0.0], [0.0, 0.95, 0.01], [-0.01, 0.0, 1.1]]);
    let mut current_coordinates = apply(&deformation_gradient, reference_coordinates);
    current_coordinates[4] += crate::mechanics::Displacement::from([0.03, -0.02, 0.015]);
    current_coordinates
}

#[test]
fn nodal_forces_and_stiffnesses_finite_difference()
-> Result<(), crate::math::assert::AssertionError> {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let block = Block::from((constitutive_model(), connectivity, &reference_coordinates));
    let coordinates = non_uniformly_deformed_coordinates(&reference_coordinates);
    let nodal_stiffnesses = block.nodal_stiffnesses(&coordinates).unwrap();
    let number_of_nodes = reference_coordinates.len();
    let mut finite_difference = NodalStiffnessesSolid::<3>::zero(number_of_nodes);
    (0..number_of_nodes).for_each(|node_b| {
        (0..3).for_each(|j| {
            let mut perturbed = coordinates.clone();
            perturbed[node_b][j] += perturbation::<Length>(0.5 * EPSILON);
            let forces_plus = block.nodal_forces(&perturbed).unwrap();
            perturbed[node_b][j] -= perturbation::<Length>(EPSILON);
            let forces_minus = block.nodal_forces(&perturbed).unwrap();
            (0..number_of_nodes).for_each(|node_a| {
                (0..3).for_each(|i| {
                    finite_difference[node_a][node_b][i][j] = (forces_plus[node_a][i]
                        - forces_minus[node_a][i])
                        / perturbation::<Length>(EPSILON);
                })
            })
        })
    });
    Assert::default().eq_within_fd_tol(&nodal_stiffnesses, &finite_difference)
}
