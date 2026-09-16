macro_rules! test_block_elastic_and_hyperelastic {
    ($element: ident) => {
        mod block {
            use super::*;
            use crate::{
                EPSILON,
                math::{Rank2, TensorArray, TensorRank2, assert::AssertionError},
                mechanics::test::{
                    get_rotation_current_configuration, get_rotation_reference_configuration,
                    get_translation_current_configuration, get_translation_reference_configuration,
                },
            };
            mod elastic {
                use super::*;
                #[allow(unused_imports)]
                use crate::{
                    constitutive::solid::elastic::{
                        AlmansiHamelEulerian, AlmansiHamelLagrangian, SaintVenantKirchhoff,
                        test::{BULK_MODULUS, SHEAR_MODULUS},
                    },
                    domain::solid::{SolidElements, elastic::ElasticElements},
                };
                mod almansi_hamel_eulerian {
                    use super::*;
                    test_finite_element_block_with_elastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamelEulerian {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        AlmansiHamelEulerian
                    );
                }
                mod almansi_hamel_lagrangian {
                    use super::*;
                    test_finite_element_block_with_elastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamelLagrangian {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        AlmansiHamelLagrangian
                    );
                }
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_block_with_elastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoff {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        SaintVenantKirchhoff
                    );
                }
            }
            mod hyperelastic {
                use super::*;
                #[allow(unused_imports)]
                use crate::{
                    constitutive::solid::hyperelastic::{
                        ArrudaBoyce, Carroll, Fung, Gent, Isihara, MooneyRivlin, NeoHookean,
                        SaintVenantKirchhoff, Yeoh,
                        test::{
                            EXPONENT, EXTENSIBILITY, EXTRA_MODULUS, LINEAR_MODULUS,
                            NUMBER_OF_LINKS, QUADRATIC_MODULUS, QUARTIC_MODULUS,
                            SECOND_INVARIANT_MODULUS, YEOH_MODULI,
                        },
                    },
                    domain::solid::{
                        SolidElements, elastic::ElasticElements, hyperelastic::HyperelasticElements,
                    },
                };
                mod arruda_boyce {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        ArrudaBoyce {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            number_of_links: NUMBER_OF_LINKS,
                        },
                        ArrudaBoyce
                    );
                }
                mod carroll {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Carroll {
                            bulk_modulus: BULK_MODULUS,
                            linear_modulus: LINEAR_MODULUS,
                            quartic_modulus: QUARTIC_MODULUS,
                            second_invariant_modulus: SECOND_INVARIANT_MODULUS,
                        },
                        Carroll
                    );
                }
                mod fung {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Fung {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            exponent: EXPONENT,
                            extra_modulus: EXTRA_MODULUS,
                        },
                        Fung
                    );
                }
                mod gent {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Gent {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extensibility: EXTENSIBILITY,
                        },
                        Gent
                    );
                }
                mod isihara {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Isihara {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extra_modulus: EXTRA_MODULUS,
                            quadratic_modulus: QUADRATIC_MODULUS,
                        },
                        Isihara
                    );
                }
                mod mooney_rivlin {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        MooneyRivlin {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extra_modulus: EXTRA_MODULUS,
                        },
                        MooneyRivlin
                    );
                }
                mod neo_hookean {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        NeoHookean {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        NeoHookean
                    );
                }
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoff {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        SaintVenantKirchhoff
                    );
                }
                mod yeoh {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Yeoh {
                            bulk_modulus: BULK_MODULUS,
                            shear_moduli: YEOH_MODULI.to_vec(),
                        },
                        Yeoh
                    );
                }
            }
        }
    };
}
pub(crate) use test_block_elastic_and_hyperelastic;

macro_rules! test_nodal_forces_and_nodal_stiffnesses {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        setup_block!($constitutive_model, $constitutive_model_type);
        fn get_coordinates_transformed_block() -> NodalCoordinates<3> {
            get_coordinates_block()
                .iter()
                .map(|coordinate| {
                    (get_rotation_current_configuration() * coordinate)
                        + get_translation_current_configuration()
                })
                .collect()
        }
        fn get_reference_coordinates_transformed_block() -> NodalReferenceCoordinates<3> {
            get_reference_coordinates_block()
                .iter()
                .map(|reference_coordinate| {
                    (get_rotation_reference_configuration() * reference_coordinate)
                        + get_translation_reference_configuration()
                })
                .collect()
        }
        mod nodal_forces {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_nodal_stiffnesses(true, false)?,
                        &get_finite_difference_of_nodal_forces(true)?,
                    )
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_nodal_forces(true, false)?,
                        &get_nodal_forces(true, true)?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_nodal_stiffnesses(false, false)?,
                        &get_finite_difference_of_nodal_forces(false)?,
                    )
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default()
                        .eq_within_tols(&get_nodal_forces(false, true)?, &NodalForcesSolid::zero(D))
                }
                #[test]
                fn zero() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_nodal_forces(false, false)?,
                        &NodalForcesSolid::zero(D),
                    )
                }
            }
        }
        mod nodal_stiffnesses {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_nodal_stiffnesses(true, false)?,
                        &get_nodal_stiffnesses(true, true)?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_nodal_stiffnesses(false, false)?,
                        &get_nodal_stiffnesses(false, true)?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_nodal_forces_and_nodal_stiffnesses;

macro_rules! test_helmholtz_free_energy {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_finite_difference_of_helmholtz_free_energy(
            is_deformed: bool,
        ) -> Result<NodalForcesSolid<3>, AssertionError> {
            let block = get_block();
            let mut finite_difference = $crate::math::Quantity::default();
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_coordinates = if is_deformed {
                                get_coordinates_block()
                            } else {
                                get_reference_coordinates_block().into()
                            };
                            nodal_coordinates[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference = block.helmholtz_free_energy(&nodal_coordinates)?;
                            nodal_coordinates = if is_deformed {
                                get_coordinates_block()
                            } else {
                                get_reference_coordinates_block().into()
                            };
                            nodal_coordinates[node][i] -=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            finite_difference -= block.helmholtz_free_energy(&nodal_coordinates)?;
                            Ok((finite_difference
                                / $crate::math::Quantity::<$crate::units::Length>::new(EPSILON))
                            .value_as::<$crate::units::Force>())
                        })
                        .collect()
                })
                .collect()
        }
        mod helmholtz_free_energy {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_block().nodal_forces(&get_coordinates_block())?,
                        &get_finite_difference_of_helmholtz_free_energy(true)?,
                    )
                }
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] = $crate::math::Quantity::new(0.0);
                    let coordinates_block = get_reference_coordinates_block()
                        .iter()
                        .map(|reference_coordinates| &deformation_gradient * reference_coordinates)
                        .collect();
                    get_block()
                        .helmholtz_free_energy(&coordinates_block)
                        .unwrap();
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_forces = block.nodal_forces(&nodal_coordinates)?;
                    let minimum = block.helmholtz_free_energy(&nodal_coordinates)?
                        - $crate::math::ContractWith::contract_with(
                            &nodal_forces,
                            &nodal_coordinates,
                        );
                    let mut perturbed = $crate::math::Quantity::default();
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_coordinates = nodal_coordinates.clone();
                            perturbed_coordinates[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - $crate::math::ContractWith::contract_with(
                                    &nodal_forces,
                                    &perturbed_coordinates,
                                );
                            if $crate::math::assert::Assert::default()
                                .eq_within_tols(&perturbed, &minimum)
                                .is_err()
                            {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - $crate::math::ContractWith::contract_with(
                                    &nodal_forces,
                                    &perturbed_coordinates,
                                );
                            if $crate::math::assert::Assert::default()
                                .eq_within_tols(&perturbed, &minimum)
                                .is_err()
                            {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &get_block().helmholtz_free_energy(&get_coordinates_block())?,
                        &get_block_transformed()
                            .helmholtz_free_energy(&get_coordinates_transformed_block())?,
                    )
                }
                #[test]
                fn positive() -> Result<(), AssertionError> {
                    let block = get_block();
                    $crate::math::assert::Assert::default().zero_within_tols(
                        &block
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?
                            .abs(),
                    )?;
                    assert!(
                        block.helmholtz_free_energy(&get_coordinates_block())?
                            > $crate::math::Quantity::default()
                    );
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &get_finite_difference_of_helmholtz_free_energy(false)?,
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), AssertionError> {
                    let mut perturbed = $crate::math::Quantity::default();
                    let mut perturbed_coordinates = get_reference_coordinates_block().into();
                    let block = get_block();
                    let minimum = block.helmholtz_free_energy(&perturbed_coordinates)?;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_coordinates = get_reference_coordinates_block().into();
                            perturbed_coordinates[node][i] +=
                                $crate::math::assert::perturbation(0.5 * EPSILON);
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?;
                            if $crate::math::assert::Assert::default()
                                .eq_within_tols(&perturbed, &minimum)
                                .is_err()
                            {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -=
                                $crate::math::assert::perturbation(EPSILON);
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?;
                            if $crate::math::assert::Assert::default()
                                .eq_within_tols(&perturbed, &minimum)
                                .is_err()
                            {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), AssertionError> {
                    let block_1 = get_block();
                    let block_2 = get_block_transformed();
                    $crate::math::assert::Assert::default().eq_within_tols(
                        &block_1
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?,
                        &block_2.helmholtz_free_energy(
                            &get_reference_coordinates_transformed_block().into(),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), AssertionError> {
                    $crate::math::assert::Assert::default().zero_within_tols(
                        &get_block()
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?
                            .abs(),
                    )
                }
            }
        }
        #[test]
        fn nodal_stiffnesses_deformed_symmetry() -> Result<(), AssertionError> {
            let nodal_stiffness = get_nodal_stiffnesses(true, false)?;
            nodal_stiffness
                .iter()
                .enumerate()
                .try_for_each(|(a, nodal_stiffness_a)| {
                    nodal_stiffness_a
                        .entries()
                        .try_for_each(|(b, nodal_stiffness_ab)| {
                            nodal_stiffness_ab.iter().enumerate().try_for_each(
                                |(i, nodal_stiffness_ab_i)| {
                                    nodal_stiffness_ab_i.iter().enumerate().try_for_each(
                                        |(j, nodal_stiffness_ab_ij)| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                nodal_stiffness_ab_ij,
                                                &nodal_stiffness[b][a][j][i],
                                            )
                                        },
                                    )
                                },
                            )
                        })
                })
        }
        #[test]
        fn nodal_stiffnesses_undeformed_symmetry() -> Result<(), AssertionError> {
            let nodal_stiffness = get_nodal_stiffnesses(false, false)?;
            nodal_stiffness
                .iter()
                .enumerate()
                .try_for_each(|(a, nodal_stiffness_a)| {
                    nodal_stiffness_a
                        .entries()
                        .try_for_each(|(b, nodal_stiffness_ab)| {
                            nodal_stiffness_ab.iter().enumerate().try_for_each(
                                |(i, nodal_stiffness_ab_i)| {
                                    nodal_stiffness_ab_i.iter().enumerate().try_for_each(
                                        |(j, nodal_stiffness_ab_ij)| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                nodal_stiffness_ab_ij,
                                                &nodal_stiffness[b][a][j][i],
                                            )
                                        },
                                    )
                                },
                            )
                        })
                })
        }
    };
}
pub(crate) use test_helmholtz_free_energy;

macro_rules! test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<NodalStiffnessesSolid<3>, AssertionError> {
            let block = get_block();
            let mut finite_difference = $crate::math::Quantity::default();
            (0..D)
                .map(|node_a| {
                    (0..D)
                        .map(|node_b| {
                            (0..3)
                                .map(|i| {
                                    (0..3)
                                        .map(|j| {
                                            let mut nodal_coordinates = if is_deformed {
                                                get_coordinates_block()
                                            } else {
                                                get_reference_coordinates_block().into()
                                            };
                                            nodal_coordinates[node_b][j] +=
                                                $crate::math::assert::perturbation(0.5 * EPSILON);
                                            finite_difference =
                                                block.nodal_forces(&nodal_coordinates)?[node_a][i];
                                            nodal_coordinates = if is_deformed {
                                                get_coordinates_block()
                                            } else {
                                                get_reference_coordinates_block().into()
                                            };
                                            nodal_coordinates[node_b][j] -=
                                                $crate::math::assert::perturbation(0.5 * EPSILON);
                                            finite_difference -=
                                                block.nodal_forces(&nodal_coordinates)?[node_a][i];
                                            Ok(finite_difference
                                                / $crate::math::assert::perturbation::<
                                                    $crate::units::Length,
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
        fn get_nodal_forces(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<NodalForcesSolid<3>, AssertionError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed()
                            .nodal_forces(&get_coordinates_transformed_block())?)
                } else {
                    let converted: TensorRank2<3, $crate::math::Current, $crate::math::Current> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed()
                            .nodal_forces(&get_reference_coordinates_transformed_block().into())?)
                }
            } else {
                if is_deformed {
                    Ok(get_block().nodal_forces(&get_coordinates_block())?)
                } else {
                    Ok(get_block().nodal_forces(&get_reference_coordinates_block().into())?)
                }
            }
        }
        fn get_nodal_stiffnesses(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<NodalStiffnessesSolid<3>, AssertionError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed()
                            .nodal_stiffnesses(&get_coordinates_transformed_block())?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, $crate::math::Current, $crate::math::Current> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed().nodal_stiffnesses(
                            &get_reference_coordinates_transformed_block().into(),
                        )?
                        * converted)
                }
            } else {
                if is_deformed {
                    Ok(get_block().nodal_stiffnesses(&get_coordinates_block())?)
                } else {
                    Ok(get_block().nodal_stiffnesses(&get_reference_coordinates_block().into())?)
                }
            }
        }
        crate::domain::block::test::test_nodal_forces_and_nodal_stiffnesses!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model;

macro_rules! test_finite_element_block_with_elastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::domain::block::test::test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        #[test]
        fn nodal_stiffnesses_deformed_non_symmetry() -> Result<(), AssertionError> {
            let nodal_stiffness = get_nodal_stiffnesses(true, false)?;
            let mut transposed = nodal_stiffness.clone();
            nodal_stiffness.iter().enumerate().for_each(|(a, row)| {
                row.entries().for_each(|(b, block)| {
                    (0..3).for_each(|i| {
                        (0..3).for_each(|j| transposed[b][a][j][i] = block[i][j])
                    })
                })
            });
            assert!($crate::math::assert::Assert::default().eq_within_tols(&nodal_stiffness, &transposed).is_err());
            Ok(())
        }
    };
}
pub(crate) use test_finite_element_block_with_elastic_constitutive_model;

macro_rules! test_finite_element_block_with_hyperelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::domain::block::test::test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        crate::domain::block::test::test_helmholtz_free_energy!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_block_with_hyperelastic_constitutive_model;
