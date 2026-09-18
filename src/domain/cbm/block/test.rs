use super::{Block, node::Weighting};
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

#[test]
fn patch_test_uniform_deformation_gradient_solid_angle_weighting() {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let block = Block::from((
        (),
        connectivity,
        &reference_coordinates,
        Weighting::SolidAngle,
    ));
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

#[test]
fn nodal_forces_and_stiffnesses_finite_difference_solid_angle_weighting()
-> Result<(), crate::math::assert::AssertionError> {
    let (connectivity, reference_coordinates) = two_tetrahedra_reference();
    let block = Block::from((
        constitutive_model(),
        connectivity,
        &reference_coordinates,
        Weighting::SolidAngle,
    ));
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

mod shared_battery {
    use super::Block;
    use crate::{
        constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        domain::{
            NodalCoordinates, NodalReferenceCoordinates,
            block::test::{
                test_finite_element_block_with_elastic_constitutive_model,
                test_finite_element_block_with_hyperelastic_constitutive_model,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
        },
        geometry::mesh::PrimitiveConnectivity,
        math::Tensor,
        mechanics::DeformationGradient,
    };

    const D: usize = 14;

    fn get_connectivity() -> PrimitiveConnectivity<3, 4> {
        vec![
            [13, 12, 8, 1],
            [10, 3, 0, 8],
            [11, 10, 8, 3],
            [12, 11, 8, 2],
            [11, 2, 3, 8],
            [12, 2, 8, 1],
            [13, 10, 5, 0],
            [13, 11, 10, 8],
            [10, 6, 9, 5],
            [12, 7, 4, 9],
            [12, 11, 7, 9],
            [11, 7, 9, 6],
            [13, 1, 8, 0],
            [13, 9, 4, 5],
            [13, 12, 1, 4],
            [11, 10, 6, 9],
            [11, 10, 3, 6],
            [12, 11, 2, 7],
            [13, 11, 9, 10],
            [13, 12, 4, 9],
            [13, 10, 0, 8],
            [13, 10, 9, 5],
            [13, 12, 11, 8],
            [13, 12, 9, 11],
        ]
        .into()
    }

    fn get_coordinates_block() -> NodalCoordinates<3> {
        NodalCoordinates::from([
            [0.48419081, -0.52698494, 0.42026988],
            [0.43559430, 0.52696224, 0.54477963],
            [-0.56594965, 0.57076191, 0.51683869],
            [-0.56061746, -0.42795457, 0.55275658],
            [0.41878700, 0.53190268, -0.44744274],
            [0.47232357, -0.57252738, -0.42946606],
            [-0.45168197, -0.5102938, -0.57959825],
            [-0.41776733, 0.41581785, -0.45911886],
            [0.05946988, 0.03773822, 0.44149305],
            [-0.08478334, -0.09009810, -0.46105872],
            [-0.04039882, -0.58201398, 0.09346960],
            [-0.57820738, 0.08325131, 0.03614415],
            [-0.04145077, 0.56406301, 0.09988905],
            [0.52149656, -0.08553510, -0.03187069],
        ])
    }

    fn get_reference_coordinates_block() -> NodalReferenceCoordinates<3> {
        NodalReferenceCoordinates::from([
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
            [0.0, -0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ])
    }

    mod uniform {
        use super::*;
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type> {
                    Block::<$constitutive_model_type>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type> {
                    Block::<$constitutive_model_type>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                    ))
                }
            };
        }

        crate::domain::block::test::test_block_elastic_and_hyperelastic!(Particle);
    }

    mod solid_angle {
        use super::*;
        use crate::cbm::Weighting;
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type> {
                    Block::<$constitutive_model_type>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                        Weighting::SolidAngle,
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type> {
                    Block::<$constitutive_model_type>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                        Weighting::SolidAngle,
                    ))
                }
            };
        }

        crate::domain::block::test::test_block_elastic_and_hyperelastic!(Particle);
    }
}
