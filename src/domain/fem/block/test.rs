macro_rules! test_finite_element_block {
    ($element: ident) => {
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type, $element, G, M, N, P> {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                    ))
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
    };
}
pub(crate) use test_finite_element_block;

macro_rules! test_surface_finite_element_block {
    ($element: ident) => {
        use crate::fem::block::element::test::THICKNESS;
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type, $element, G, M, N, P> {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                        THICKNESS,
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                        THICKNESS,
                    ))
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
    };
}
pub(crate) use test_surface_finite_element_block;

macro_rules! test_finite_element_block_inner {
    ($element: ident) => {
        macro_rules! test_finite_element_block_with_elastic_constitutive_model {
            ($block: ident, $element_: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
                crate::domain::block::test::test_finite_element_block_with_elastic_constitutive_model!(
                    $block,
                    $element_,
                    $constitutive_model,
                    $constitutive_model_type
                );
                macro_rules! test_root_with_solver {
                    ($solver: ident) => {
                        #[test]
                        fn root() -> Result<(), AssertionError> {
                            let (applied_load, a, b) = equality_constraint();
                            let block = get_block();
                            let coordinates = FirstOrderRoot::root(
                                &crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )),
                                EqualityConstraint::Linear(a, b),
                                $solver::default(),
                            )?;
                            let deformation_gradient =
                                $constitutive_model.root(applied_load, $solver::default())?;
                            block
                                .deformation_gradients(&coordinates)
                                .iter()
                                .try_for_each(|deformation_gradients| {
                                    deformation_gradients.iter().try_for_each(
                                        |deformation_gradient_g| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                deformation_gradient_g,
                                                &deformation_gradient,
                                            )
                                        },
                                    )
                                })
                        }
                    };
                }
                mod newton_raphson_root {
                    use super::*;
                    use crate::{
                        constitutive::solid::elastic::FirstOrderRoot as _, domain::FirstOrderRoot,
                        math::optimize::NewtonRaphson,
                    };
                    test_root_with_solver!(NewtonRaphson);
                }
            };
        }
        macro_rules! test_finite_element_block_with_hyperelastic_constitutive_model {
            ($block: ident, $element_: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
                crate::domain::block::test::test_finite_element_block_with_hyperelastic_constitutive_model!(
                    $block,
                    $element_,
                    $constitutive_model,
                    $constitutive_model_type
                );
                macro_rules! test_minimize_with_solver {
                    ($solver: ident) => {
                        #[test]
                        fn minimize() -> Result<(), AssertionError> {
                            let (applied_load, a, b) = equality_constraint();
                            let block = get_block();
                            let coordinates = SecondOrderMinimize::minimize(
                                &crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )),
                                EqualityConstraint::Linear(a, b),
                                $solver::default(),
                            )?;
                            let deformation_gradient =
                                $constitutive_model.minimize(applied_load, $solver::default())?;
                            block
                                .deformation_gradients(&coordinates)
                                .iter()
                                .try_for_each(|deformation_gradients| {
                                    deformation_gradients.iter().try_for_each(
                                        |deformation_gradient_g| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                deformation_gradient_g,
                                                &deformation_gradient,
                                            )
                                        },
                                    )
                                })
                        }
                    };
                }
                mod newton_raphson_minimize {
                    use super::*;
                    use crate::{
                        constitutive::solid::hyperelastic::SecondOrderMinimize as _,
                        domain::SecondOrderMinimize, math::optimize::NewtonRaphson,
                    };
                    test_minimize_with_solver!(NewtonRaphson);
                }
            };
        }
        crate::domain::block::test::test_block_elastic_and_hyperelastic!($element);
        mod block_viscous {
            use super::*;
            use crate::{
                EPSILON,
                fem::block::test::{
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model,
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model,
                },
                math::{Rank2, TensorRank2, assert::AssertionError},
                mechanics::test::{
                    get_rotation_current_configuration, get_rotation_rate_current_configuration,
                    get_rotation_reference_configuration, get_translation_current_configuration,
                    get_translation_rate_current_configuration,
                    get_translation_reference_configuration,
                },
            };
            mod elastic_hyperviscous {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical,
                        fluid::hyperviscous::Newtonian,
                        solid::{
                            elastic::AlmansiHamelEulerian,
                            elastic_hyperviscous::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                        },
                    },
                    fem::{
                        block::solid::SolidElements,
                        solid::{
                            NodalDampingsSolid, elastic_hyperviscous::ElasticHyperviscousElements,
                            viscoelastic::ViscoelasticElements,
                        },
                    },
                };
                type AlmansiHamel = Canonical<AlmansiHamelEulerian, Newtonian>;
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamel::from((
                            AlmansiHamelEulerian {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            Newtonian {
                                bulk_viscosity: BULK_VISCOSITY,
                                shear_viscosity: SHEAR_VISCOSITY,
                            },
                        )),
                        AlmansiHamel
                    );
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
                    fem::{
                        block::solid::SolidElements,
                        solid::{
                            NodalDampingsSolid, elastic_hyperviscous::ElasticHyperviscousElements,
                            viscoelastic::ViscoelasticElements,
                        },
                    },
                };
                type SaintVenantKirchhoff =
                    Canonical<HyperelasticSaintVenantKirchhoff, ViscousSaintVenantKirchhoff>;
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoff::from((
                            HyperelasticSaintVenantKirchhoff {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            ViscousSaintVenantKirchhoff {
                                bulk_viscosity: BULK_VISCOSITY,
                                shear_viscosity: SHEAR_VISCOSITY,
                            },
                        )),
                        SaintVenantKirchhoff
                    );
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_inner;

macro_rules! test_finite_element_block_with_viscoelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        use crate::math::ContractWith;
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
        fn get_nodal_forces(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<NodalForcesSolid<3>, AssertionError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed().nodal_forces(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?)
                } else {
                    let converted: TensorRank2<3, $crate::math::Current, $crate::math::Current> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed().nodal_forces(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?)
                }
            } else {
                if is_deformed {
                    Ok(get_block()
                        .nodal_forces(&get_coordinates_block(), &get_velocities_block())?)
                } else {
                    Ok(get_block().nodal_forces(
                        &get_reference_coordinates_block().into(),
                        &NodalVelocities::zero(D),
                    )?)
                }
            }
        }
        fn get_nodal_stiffnesses(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<NodalDampingsSolid<3>, AssertionError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed().nodal_stiffnesses(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, $crate::math::Current, $crate::math::Current> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed().nodal_stiffnesses(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?
                        * converted)
                }
            } else {
                if is_deformed {
                    Ok(get_block()
                        .nodal_stiffnesses(&get_coordinates_block(), &get_velocities_block())?)
                } else {
                    Ok(get_block().nodal_stiffnesses(
                        &get_reference_coordinates_block().into(),
                        &NodalVelocities::zero(D),
                    )?)
                }
            }
        }
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<NodalDampingsSolid<3>, AssertionError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = $crate::math::Quantity::default();
            (0..D)
                .map(|node_a| {
                    (0..D)
                        .map(|node_b| {
                            (0..3)
                                .map(|i| {
                                    (0..3)
                                        .map(|j| {
                                            let mut nodal_velocities = if is_deformed {
                                                get_velocities_block()
                                            } else {
                                                NodalVelocities::zero(D)
                                            };
                                            nodal_velocities[node_a][i] +=
                                                $crate::math::assert::perturbation(0.5 * EPSILON);
                                            finite_difference = block.nodal_forces(
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[node_b][j];
                                            nodal_velocities = if is_deformed {
                                                get_velocities_block()
                                            } else {
                                                NodalVelocities::zero(D)
                                            };
                                            nodal_velocities[node_a][i] -=
                                                $crate::math::assert::perturbation(0.5 * EPSILON);
                                            finite_difference -= block.nodal_forces(
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[node_b][j];
                                            Ok(finite_difference
                                                / $crate::math::assert::perturbation::<
                                                    $crate::units::Velocity,
                                                >(EPSILON))
                                        })
                                        .collect()
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        }
        crate::domain::block::test::test_nodal_forces_and_nodal_stiffnesses!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_block_with_viscoelastic_constitutive_model;

macro_rules! test_finite_element_block_with_elastic_hyperviscous_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::test::test_finite_element_block_with_viscoelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        use crate::math::{
            Quantity,
            integrate::{BogackiShampine, DormandPrince, Verner8, Verner9},
            optimize::NewtonRaphson,
        };
        macro_rules! test_with_integrator {
            ($integrator: ident) => {
                #[test]
                fn minimize() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::elastic_hyperviscous::SecondOrderMinimize as _;
                    use crate::fem::solid::elastic_hyperviscous::SecondOrderMinimize;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) =
                        SecondOrderMinimize::minimize(
                            &crate::fem::Model::from((
                                get_block(),
                                get_reference_coordinates_block(),
                            )),
                            EqualityConstraint::Linear(a, b),
                            $integrator::default(),
                            &[Quantity::new(0.0), Quantity::new(1.0)],
                            NewtonRaphson::default(),
                        )?;
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model.minimize(
                            applied_velocity(&times),
                            $integrator::default(),
                            NewtonRaphson::default(),
                        )?;
                    coordinates_history
                        .iter()
                        .zip(
                            velocities_history.iter().zip(
                                deformation_gradients
                                    .iter()
                                    .zip(deformation_gradient_rates.iter()),
                            ),
                        )
                        .try_for_each(
                            |(
                                coordinates,
                                (velocities, (deformation_gradient, deformation_gradient_rate)),
                            )| {
                                block
                                    .deformation_gradients(coordinates)
                                    .iter()
                                    .try_for_each(|deformation_gradients| {
                                        deformation_gradients.iter().try_for_each(
                                            |deformation_gradient_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_g,
                                                        deformation_gradient,
                                                    )
                                            },
                                        )
                                    })?;
                                block
                                    .deformation_gradient_rates(coordinates, velocities)
                                    .iter()
                                    .try_for_each(|deformation_gradient_rates| {
                                        deformation_gradient_rates.iter().try_for_each(
                                            |deformation_gradient_rate_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_rate_g,
                                                        deformation_gradient_rate,
                                                    )
                                            },
                                        )
                                    })
                            },
                        )
                }
                #[test]
                fn root() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::viscoelastic::FirstOrderRoot as _;
                    use crate::fem::solid::viscoelastic::FirstOrderRoot;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) = FirstOrderRoot::root(
                        &crate::fem::Model::from((get_block(), get_reference_coordinates_block())),
                        EqualityConstraint::Linear(a, b),
                        $integrator::default(),
                        &[Quantity::new(0.0), Quantity::new(1.0)],
                        NewtonRaphson::default(),
                    )?;
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model.root(
                            applied_velocity(&times),
                            $integrator::default(),
                            NewtonRaphson::default(),
                        )?;
                    coordinates_history
                        .iter()
                        .zip(
                            velocities_history.iter().zip(
                                deformation_gradients
                                    .iter()
                                    .zip(deformation_gradient_rates.iter()),
                            ),
                        )
                        .try_for_each(
                            |(
                                coordinates,
                                (velocities, (deformation_gradient, deformation_gradient_rate)),
                            )| {
                                block
                                    .deformation_gradients(coordinates)
                                    .iter()
                                    .try_for_each(|deformation_gradients| {
                                        deformation_gradients.iter().try_for_each(
                                            |deformation_gradient_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_g,
                                                        deformation_gradient,
                                                    )
                                            },
                                        )
                                    })?;
                                block
                                    .deformation_gradient_rates(coordinates, velocities)
                                    .iter()
                                    .try_for_each(|deformation_gradient_rates| {
                                        deformation_gradient_rates.iter().try_for_each(
                                            |deformation_gradient_rate_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_rate_g,
                                                        deformation_gradient_rate,
                                                    )
                                            },
                                        )
                                    })
                            },
                        )
                }
            };
        }
        mod bogacki_shampine {
            use super::*;
            test_with_integrator!(BogackiShampine);
        }
        mod dormand_prince {
            use super::*;
            test_with_integrator!(DormandPrince);
        }
        mod verner_8 {
            use super::*;
            test_with_integrator!(Verner8);
        }
        mod verner_9 {
            use super::*;
            test_with_integrator!(Verner9);
        }
        fn get_finite_difference_of_viscous_dissipation(
            is_deformed: bool,
        ) -> Result<NodalForcesSolid<3>, AssertionError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = $crate::math::Quantity::default();
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference =
                                block.viscous_dissipation(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] -=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference -=
                                block.viscous_dissipation(&nodal_coordinates, &nodal_velocities)?;
                            Ok((finite_difference
                                / $crate::math::Quantity::<$crate::units::Velocity>::new(EPSILON))
                            .value_as::<$crate::units::Force>())
                        })
                        .collect()
                })
                .collect()
        }
        fn get_finite_difference_of_dissipation_potential(
            is_deformed: bool,
        ) -> Result<NodalForcesSolid<3>, AssertionError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = $crate::math::Quantity::default();
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference = block
                                .dissipation_potential(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] -=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference -= block
                                .dissipation_potential(&nodal_coordinates, &nodal_velocities)?;
                            Ok((finite_difference
                                / $crate::math::Quantity::<$crate::units::Velocity>::new(EPSILON))
                            .value_as::<$crate::units::Force>())
                        })
                        .collect()
                })
                .collect()
        }
        mod viscous_dissipation {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_forces_0 =
                        block.nodal_forces(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &(block.nodal_forces(&nodal_coordinates, &get_velocities_block())?
                            - nodal_forces_0),
                        &get_finite_difference_of_viscous_dissipation(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_velocities = get_velocities_block();
                    let nodal_forces_0 =
                        block.nodal_forces(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    let nodal_forces =
                        block.nodal_forces(&nodal_coordinates, &nodal_velocities)? - nodal_forces_0;
                    let minimum = block
                        .viscous_dissipation(&nodal_coordinates, &nodal_velocities)?
                        - nodal_forces.contract_with(&nodal_velocities);
                    let mut perturbed_velocities = get_velocities_block();
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = get_velocities_block();
                            perturbed_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.contract_with(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.contract_with(&perturbed_velocities)
                                    >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_block().viscous_dissipation(
                            &get_coordinates_block(),
                            &get_velocities_block(),
                        )?,
                        &get_block_transformed().viscous_dissipation(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?,
                    )
                }
                #[test]
                fn positive() -> Result<(), AssertionError> {
                    assert!(
                        get_block().viscous_dissipation(
                            &get_coordinates_block(),
                            &get_velocities_block(),
                        )? > $crate::math::Quantity::default()
                    );
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_finite_difference_of_viscous_dissipation(false)?,
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum =
                        block.viscous_dissipation(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    let mut perturbed_velocities = NodalVelocities::zero(D);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = NodalVelocities::zero(D);
                            perturbed_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            perturbed_velocities[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_block().viscous_dissipation(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                        &get_block_transformed().viscous_dissipation(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::zero(&get_block().viscous_dissipation(
                        &get_reference_coordinates_block().into(),
                        &NodalVelocities::zero(D),
                    )?)
                }
            }
        }
        mod dissipation_potential {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_block()
                            .nodal_forces(&get_coordinates_block(), &get_velocities_block())?,
                        &get_finite_difference_of_dissipation_potential(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_velocities = get_velocities_block();
                    let nodal_forces = block.nodal_forces(&nodal_coordinates, &nodal_velocities)?;
                    let minimum = block
                        .dissipation_potential(&nodal_coordinates, &nodal_velocities)?
                        - nodal_forces.contract_with(&nodal_velocities);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = nodal_velocities.clone();
                            perturbed_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.contract_with(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.contract_with(&perturbed_velocities)
                                    >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_block().dissipation_potential(
                            &get_coordinates_block(),
                            &get_velocities_block(),
                        )?,
                        &get_block_transformed().dissipation_potential(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_finite_difference_of_dissipation_potential(false)?,
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum = block
                        .dissipation_potential(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = NodalVelocities::zero(D);
                            perturbed_velocities[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            perturbed_velocities[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_block().dissipation_potential(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                        &get_block_transformed().dissipation_potential(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::zero(&get_block().dissipation_potential(
                        &get_reference_coordinates_block().into(),
                        &NodalVelocities::zero(D),
                    )?)
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_with_elastic_hyperviscous_constitutive_model;

macro_rules! test_finite_element_block_with_hyperviscoelastic_constitutive_model
{
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) =>
    {
        crate::fem::block::test::test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
            $block, $element, $constitutive_model, $constitutive_model_type
        );
        // crate::fem::block::test::test_helmholtz_free_energy!($block, $element, $constitutive_model, $constitutive_model_type);
        #[test]
        fn dissipation_potential_deformed_positive() -> Result<(), AssertionError>
        {
            assert!(
                get_block().dissipation_potential(
                    &get_coordinates_block(),
                    &get_velocities_block()
                )? > $crate::math::Quantity::default()
            );
            Ok(())
        }
    }
}
pub(crate) use test_finite_element_block_with_hyperviscoelastic_constitutive_model;
