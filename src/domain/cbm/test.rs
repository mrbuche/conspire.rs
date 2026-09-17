use super::Cbm;
use crate::{
    cbm::SolidElements,
    domain::{NodalCoordinates, NodalReferenceCoordinates, NodalVelocities, nodal_coordinates},
    geometry::{Coordinate, Coordinates, mesh::PrimitiveConnectivity},
    math::{Tensor, assert::Assert},
    mechanics::{DeformationGradient, DeformationGradientRate},
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
    let cbm = Cbm::from((connectivity, &reference_coordinates));
    let deformation_gradient =
        DeformationGradient::from([[1.1, 0.05, 0.0], [0.0, 0.9, 0.02], [-0.03, 0.0, 1.2]]);
    let current_coordinates = apply(&deformation_gradient, &reference_coordinates);
    cbm.deformation_gradients(&current_coordinates)
        .iter()
        .try_for_each(|particle_deformation_gradient| {
            Assert::default().eq_within_tols(particle_deformation_gradient, &deformation_gradient)
        })
        .unwrap()
}

#[test]
fn patch_test_uniform_deformation_gradient_rate() {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let cbm = Cbm::from((connectivity, &reference_coordinates));
    let deformation_gradient =
        DeformationGradient::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let current_coordinates = apply(&deformation_gradient, &reference_coordinates);
    let deformation_gradient_rate =
        DeformationGradientRate::from([[0.1, 0.0, 0.0], [0.0, -0.05, 0.01], [0.0, 0.0, 0.02]]);
    let velocities: NodalVelocities<3> = reference_coordinates
        .iter()
        .map(|reference_coordinate| &deformation_gradient_rate * reference_coordinate)
        .collect();
    cbm.deformation_gradient_rates(&current_coordinates, &velocities)
        .iter()
        .try_for_each(|particle_deformation_gradient_rate| {
            Assert::default().eq_within_tols(
                particle_deformation_gradient_rate,
                &deformation_gradient_rate,
            )
        })
        .unwrap()
}
