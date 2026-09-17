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
    math::Tensor,
    mechanics::{DeformationGradient, test::get_deformation_gradient},
    vem::block::{Block, element::Element},
};

const D: usize = 20;

fn get_element_face_connectivity() -> Vec<Vec<usize>> {
    vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
}

fn get_face_node_connectivity() -> Vec<Vec<usize>> {
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

fn get_reference_coordinates_block() -> NodalReferenceCoordinates<3> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    NodalReferenceCoordinates::from([
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

fn get_coordinates_block() -> NodalCoordinates<3> {
    get_reference_coordinates_block()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient() * reference_coordinate)
        .collect()
}

macro_rules! setup_block {
    ($constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_block() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_block(),
            ))
        }
        fn get_block_transformed() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_transformed_block(),
            ))
        }
    };
}

crate::domain::block::test::test_block_elastic_and_hyperelastic!(Element);

mod viscoelastic {
    use super::*;
    use crate::{
        EPSILON,
        constitutive::{
            canonical::Canonical,
            fluid::hyperviscous::Newtonian,
            solid::{
                elastic::AlmansiHamelEulerian,
                elastic_hyperviscous::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
            },
        },
        domain::{NodalVelocities, solid::viscoelastic::ViscoelasticElements},
        math::{Rank2, assert::AssertionError},
        mechanics::test::{
            get_deformation_gradient_rate, get_rotation_current_configuration,
            get_rotation_rate_current_configuration, get_rotation_reference_configuration,
            get_translation_current_configuration, get_translation_rate_current_configuration,
            get_translation_reference_configuration,
        },
        vem::block::solid::NodalDampingsSolid,
    };

    type AlmansiHamel = Canonical<AlmansiHamelEulerian, Newtonian>;

    fn get_model() -> AlmansiHamel {
        AlmansiHamel::from((
            AlmansiHamelEulerian {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
            Newtonian {
                bulk_viscosity: BULK_VISCOSITY,
                shear_viscosity: SHEAR_VISCOSITY,
            },
        ))
    }
    fn get_block() -> Block<AlmansiHamel, Element> {
        Block::<AlmansiHamel, Element>::from((
            get_model(),
            get_element_face_connectivity(),
            get_face_node_connectivity(),
            &get_reference_coordinates_block(),
        ))
    }
    fn get_reference_coordinates_transformed_block() -> NodalReferenceCoordinates<3> {
        get_reference_coordinates_block()
            .iter()
            .map(|reference_coordinate| {
                get_rotation_reference_configuration() * reference_coordinate
                    + get_translation_reference_configuration()
            })
            .collect()
    }
    fn get_block_transformed() -> Block<AlmansiHamel, Element> {
        Block::<AlmansiHamel, Element>::from((
            get_model(),
            get_element_face_connectivity(),
            get_face_node_connectivity(),
            &get_reference_coordinates_transformed_block(),
        ))
    }
    fn get_coordinates_transformed_block() -> NodalCoordinates<3> {
        get_coordinates_block()
            .iter()
            .map(|coordinate| {
                get_rotation_current_configuration() * coordinate
                    + get_translation_current_configuration()
            })
            .collect()
    }
    fn get_velocities_block() -> NodalVelocities<3> {
        get_reference_coordinates_block()
            .iter()
            .map(|reference_coordinate| get_deformation_gradient_rate() * reference_coordinate)
            .collect()
    }
    fn get_velocities_transformed_block() -> NodalVelocities<3> {
        get_coordinates_block()
            .iter()
            .zip(get_velocities_block().iter())
            .map(|(coordinate, velocity)| {
                get_rotation_current_configuration() * velocity
                    + get_rotation_rate_current_configuration() * coordinate
                    + get_translation_rate_current_configuration()
            })
            .collect()
    }
    #[test]
    fn nodal_forces_zero() -> Result<(), AssertionError> {
        crate::math::assert::Assert::default().eq_within_tols(
            &get_block().nodal_forces(
                &get_reference_coordinates_block().into(),
                &NodalVelocities::<3>::zero(D),
            )?,
            &NodalForcesSolid::zero(D),
        )
    }
    #[test]
    fn nodal_forces_objectivity() -> Result<(), AssertionError> {
        crate::math::assert::Assert::default().eq_within_tols(
            &(get_rotation_current_configuration().transpose()
                * get_block_transformed().nodal_forces(
                    &get_coordinates_transformed_block(),
                    &get_velocities_transformed_block(),
                )?),
            &get_block().nodal_forces(&get_coordinates_block(), &get_velocities_block())?,
        )
    }
    #[test]
    fn nodal_stiffnesses_finite_difference() -> Result<(), AssertionError> {
        let block = get_block();
        let coordinates = get_coordinates_block();
        let velocities = get_velocities_block();
        let mut finite_difference = crate::math::Quantity::default();
        let nodal_stiffnesses_fd: NodalDampingsSolid = (0..D)
            .map(|node_a| {
                (0..D)
                    .map(|node_b| {
                        (0..3)
                            .map(|i| {
                                (0..3)
                                    .map(|j| {
                                        let mut perturbed_velocities = velocities.clone();
                                        perturbed_velocities[node_b][j] +=
                                            crate::math::assert::perturbation(0.5 * EPSILON);
                                        finite_difference = block
                                            .nodal_forces(&coordinates, &perturbed_velocities)?
                                            [node_a][i];
                                        perturbed_velocities[node_b][j] -=
                                            crate::math::assert::perturbation(EPSILON);
                                        finite_difference -= block
                                            .nodal_forces(&coordinates, &perturbed_velocities)?
                                            [node_a][i];
                                        Ok(finite_difference
                                            / crate::math::assert::perturbation::<
                                                crate::units::Velocity,
                                            >(EPSILON))
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect::<Result<_, AssertionError>>()?;
        crate::math::assert::Assert::default().eq_within_fd_tol(
            &block.nodal_stiffnesses(&coordinates, &velocities)?,
            &nodal_stiffnesses_fd,
        )
    }
    #[test]
    fn viscous_dissipation_positive() -> Result<(), AssertionError> {
        use crate::domain::solid::elastic_hyperviscous::ElasticHyperviscousElements;
        assert!(
            get_block().viscous_dissipation(&get_coordinates_block(), &get_velocities_block())?
                > crate::math::Quantity::default()
        );
        Ok(())
    }
    #[test]
    fn viscous_dissipation_zero() -> Result<(), AssertionError> {
        use crate::domain::solid::elastic_hyperviscous::ElasticHyperviscousElements;
        crate::math::assert::Assert::zero(
            &get_block()
                .viscous_dissipation(&get_coordinates_block(), &NodalVelocities::<3>::zero(D))?,
        )
    }
}

mod hyperviscoelastic {
    use super::*;
    use crate::{
        constitutive::{
            canonical::Canonical,
            fluid::hyperviscous::SaintVenantKirchhoff as ViscousSaintVenantKirchhoff,
            solid::{
                hyperelastic::SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
                hyperviscoelastic::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
            },
        },
        domain::{NodalVelocities, solid::hyperviscoelastic::HyperviscoelasticElements},
        math::assert::AssertionError,
        mechanics::test::get_deformation_gradient_rate,
    };

    type SaintVenantKirchhoff =
        Canonical<HyperelasticSaintVenantKirchhoff, ViscousSaintVenantKirchhoff>;

    fn get_model() -> SaintVenantKirchhoff {
        SaintVenantKirchhoff::from((
            HyperelasticSaintVenantKirchhoff {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
            ViscousSaintVenantKirchhoff {
                bulk_viscosity: BULK_VISCOSITY,
                shear_viscosity: SHEAR_VISCOSITY,
            },
        ))
    }
    fn get_block() -> Block<SaintVenantKirchhoff, Element> {
        Block::<SaintVenantKirchhoff, Element>::from((
            get_model(),
            get_element_face_connectivity(),
            get_face_node_connectivity(),
            &get_reference_coordinates_block(),
        ))
    }
    fn get_velocities_block() -> NodalVelocities<3> {
        get_reference_coordinates_block()
            .iter()
            .map(|reference_coordinate| get_deformation_gradient_rate() * reference_coordinate)
            .collect()
    }
    #[test]
    fn helmholtz_free_energy_zero() -> Result<(), AssertionError> {
        crate::math::assert::Assert::default().zero_within_tols(
            &get_block().helmholtz_free_energy(&get_reference_coordinates_block().into())?,
        )
    }
    #[test]
    fn helmholtz_free_energy_positive() -> Result<(), AssertionError> {
        assert!(
            get_block().helmholtz_free_energy(&get_coordinates_block())?
                > crate::math::Quantity::default()
        );
        Ok(())
    }
    #[test]
    fn dissipation_potential_positive() -> Result<(), AssertionError> {
        use crate::domain::solid::elastic_hyperviscous::ElasticHyperviscousElements;
        assert!(
            get_block().dissipation_potential(&get_coordinates_block(), &get_velocities_block())?
                > crate::math::Quantity::default()
        );
        Ok(())
    }
}
