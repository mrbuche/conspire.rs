macro_rules! test_finite_element_block {
    ($element: ident) => {
        macro_rules! setup_constitutive {
            ($constitutive_model: ident, $constitutive_model_parameters: ident) => {
                fn get_block<'a>() -> ElementBlock<$element<$constitutive_model<'a>>, N> {
                    ElementBlock::<$element<$constitutive_model<'a>>, N>::new(
                        $constitutive_model_parameters,
                        get_connectivity(),
                        get_reference_coordinates_block(),
                    )
                }
                fn get_block_transformed<'a>() -> ElementBlock<$element<$constitutive_model<'a>>, N>
                {
                    ElementBlock::<$element<$constitutive_model<'a>>, N>::new(
                        $constitutive_model_parameters,
                        get_connectivity(),
                        get_reference_coordinates_transformed_block(),
                    )
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
    };
}
pub(crate) use test_finite_element_block;

macro_rules! test_surface_finite_element_block {
    ($element: ident) => {
        use super::element::test::THICKNESS;
        macro_rules! setup_constitutive {
            ($constitutive_model: ident, $constitutive_model_parameters: ident) => {
                fn get_block<'a>() -> ElementBlock<$element<$constitutive_model<'a>>, N> {
                    ElementBlock::<$element<$constitutive_model<'a>>, N>::new(
                        $constitutive_model_parameters,
                        get_connectivity(),
                        get_reference_coordinates_block(),
                        THICKNESS,
                    )
                }
                fn get_block_transformed<'a>() -> ElementBlock<$element<$constitutive_model<'a>>, N>
                {
                    ElementBlock::<$element<$constitutive_model<'a>>, N>::new(
                        $constitutive_model_parameters,
                        get_connectivity(),
                        get_reference_coordinates_transformed_block(),
                        THICKNESS,
                    )
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
    };
}
pub(crate) use test_surface_finite_element_block;

macro_rules! test_finite_element_block_inner {
    ($element: ident) => {
        mod block {
            use super::*;
            use crate::{
                EPSILON,
                fem::block::test::{
                    test_finite_element_block_with_elastic_constitutive_model,
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model,
                    test_finite_element_block_with_hyperelastic_constitutive_model,
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model,
                },
                math::{
                    Rank2, TensorArray, TensorRank2,
                    test::{TestError, assert_eq, assert_eq_from_fd, assert_eq_within_tols},
                },
                mechanics::test::{
                    get_rotation_current_configuration, get_rotation_rate_current_configuration,
                    get_rotation_reference_configuration, get_translation_current_configuration,
                    get_translation_rate_current_configuration,
                    get_translation_reference_configuration,
                },
            };
            mod elastic {
                use super::*;
                use crate::constitutive::solid::elastic::{
                    AlmansiHamel, test::ALMANSIHAMELPARAMETERS,
                };
                mod almansi_hamel {
                    use super::*;
                    type AlmansiHamelType<'a> = AlmansiHamel<&'a [Scalar; 2]>;
                    test_finite_element_block_with_elastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamelType,
                        ALMANSIHAMELPARAMETERS
                    );
                }
            }
            mod hyperelastic {
                use super::*;
                use crate::constitutive::solid::hyperelastic::{
                    ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff, Yeoh,
                    test::{
                        ARRUDABOYCEPARAMETERS, FUNGPARAMETERS, GENTPARAMETERS,
                        MOONEYRIVLINPARAMETERS, NEOHOOKEANPARAMETERS,
                        SAINTVENANTKIRCHOFFPARAMETERS, YEOHPARAMETERS,
                    },
                };
                mod arruda_boyce {
                    use super::*;
                    type ArrudaBoyceType<'a> = ArrudaBoyce<&'a [Scalar; 3]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        ArrudaBoyceType,
                        ARRUDABOYCEPARAMETERS
                    );
                }
                mod fung {
                    use super::*;
                    type FungType<'a> = Fung<&'a [Scalar; 4]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        FungType,
                        FUNGPARAMETERS
                    );
                }
                mod gent {
                    use super::*;
                    type GentType<'a> = Gent<&'a [Scalar; 3]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        GentType,
                        GENTPARAMETERS
                    );
                }
                mod mooney_rivlin {
                    use super::*;
                    type MooneyRivlinType<'a> = MooneyRivlin<&'a [Scalar; 3]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        MooneyRivlinType,
                        MOONEYRIVLINPARAMETERS
                    );
                }
                mod neo_hookean {
                    use super::*;
                    type NeoHookeanType<'a> = NeoHookean<&'a [Scalar; 2]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        NeoHookeanType,
                        NEOHOOKEANPARAMETERS
                    );
                }
                mod saint_venant_kirchhoff {
                    use super::*;
                    type SaintVenantKirchhoffType<'a> = SaintVenantKirchhoff<&'a [Scalar; 2]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoffType,
                        SAINTVENANTKIRCHOFFPARAMETERS
                    );
                }
                mod yeoh {
                    use super::*;
                    type YeohType<'a> = Yeoh<&'a [Scalar; 6]>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        YeohType,
                        YEOHPARAMETERS
                    );
                }
            }
            mod elastic_hyperviscous {
                use super::*;
                use crate::constitutive::solid::elastic_hyperviscous::{
                    AlmansiHamel, test::ALMANSIHAMELPARAMETERS,
                };
                mod almansi_hamel {
                    use super::*;
                    type AlmansiHamelType<'a> = AlmansiHamel<&'a [Scalar; 4]>;
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamelType,
                        ALMANSIHAMELPARAMETERS
                    );
                }
            }
            mod hyperviscoelastic {
                use super::*;
                use crate::constitutive::solid::hyperviscoelastic::{
                    SaintVenantKirchhoff, test::SAINTVENANTKIRCHOFFPARAMETERS,
                };
                mod saint_venant_kirchhoff {
                    use super::*;
                    type SaintVenantKirchhoffType<'a> = SaintVenantKirchhoff<&'a [Scalar; 4]>;
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoffType,
                        SAINTVENANTKIRCHOFFPARAMETERS
                    );
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_inner;

macro_rules! test_nodal_forces_and_nodal_stiffnesses {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        setup_constitutive!($constitutive_model, $constitutive_model_parameters);
        fn get_coordinates_transformed_block() -> NodalCoordinatesBlock {
            get_coordinates_block()
                .iter()
                .map(|coordinate| {
                    (get_rotation_current_configuration() * coordinate)
                        + get_translation_current_configuration()
                })
                .collect()
        }
        fn get_reference_coordinates_transformed_block() -> ReferenceNodalCoordinatesBlock {
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
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_nodal_stiffnesses(true, false)?,
                        &get_finite_difference_of_nodal_forces(true)?,
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_forces(true, false)?,
                        &get_nodal_forces(true, true)?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_nodal_stiffnesses(false, false)?,
                        &get_finite_difference_of_nodal_forces(false)?,
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_forces(false, true)?,
                        &NodalForcesBlock::zero(D),
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_forces(false, false)?,
                        &NodalForcesBlock::zero(D),
                    )
                }
            }
        }
        mod nodal_stiffnesses {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_stiffnesses(true, false)?,
                        &get_nodal_stiffnesses(true, true)?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_stiffnesses(false, false)?,
                        &get_nodal_stiffnesses(false, true)?,
                    )
                }
            }
        }
        // #[test]
        // fn size() {
        //     assert_eq!(
        //         std::mem::size_of_val(&get_block()),
        //         std::mem::size_of::<Connectivity<N>>()
        //             + get_connectivity().len() * std::mem::size_of::<$element::<$constitutive_model>>()
        //     )
        // }
    };
}
pub(crate) use test_nodal_forces_and_nodal_stiffnesses;

macro_rules! test_helmholtz_free_energy {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        fn get_finite_difference_of_helmholtz_free_energy(
            is_deformed: bool,
        ) -> Result<NodalForcesBlock, TestError> {
            let block = get_block();
            let mut finite_difference = 0.0;
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_coordinates = if is_deformed {
                                get_coordinates_block()
                            } else {
                                get_reference_coordinates_block().into()
                            };
                            nodal_coordinates[node][i] += 0.5 * EPSILON;
                            finite_difference = block.helmholtz_free_energy(&nodal_coordinates)?;
                            nodal_coordinates = if is_deformed {
                                get_coordinates_block()
                            } else {
                                get_reference_coordinates_block().into()
                            };
                            nodal_coordinates[node][i] -= 0.5 * EPSILON;
                            finite_difference -= block.helmholtz_free_energy(&nodal_coordinates)?;
                            Ok(finite_difference / EPSILON)
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
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_block().nodal_forces(&get_coordinates_block())?,
                        &get_finite_difference_of_helmholtz_free_energy(true)?,
                    )
                }
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] = 0.0;
                    let coordinates_block = get_reference_coordinates_block()
                        .iter()
                        .map(|reference_coordinates| &deformation_gradient * reference_coordinates)
                        .collect();
                    get_block()
                        .helmholtz_free_energy(&coordinates_block)
                        .unwrap();
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_forces = block.nodal_forces(&nodal_coordinates)?;
                    let minimum = block.helmholtz_free_energy(&nodal_coordinates)?
                        - nodal_forces.dot(&nodal_coordinates);
                    let mut perturbed = 0.0;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_coordinates = nodal_coordinates.clone();
                            perturbed_coordinates[node][i] += 0.5 * EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - nodal_forces.dot(&perturbed_coordinates);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -= EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - nodal_forces.dot(&perturbed_coordinates);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_block().helmholtz_free_energy(&get_coordinates_block())?,
                        &get_block_transformed()
                            .helmholtz_free_energy(&get_coordinates_transformed_block())?,
                    )
                }
                #[test]
                fn positive() -> Result<(), TestError> {
                    let block = get_block();
                    assert_eq_within_tols(
                        &block
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?
                            .abs(),
                        &0.0,
                    )?;
                    assert!(block.helmholtz_free_energy(&get_coordinates_block())? > 0.0);
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_helmholtz_free_energy(false)?,
                        &NodalForcesBlock::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let mut perturbed = 0.0;
                    let mut perturbed_coordinates = get_reference_coordinates_block().into();
                    let block = get_block();
                    let minimum = block.helmholtz_free_energy(&perturbed_coordinates)?;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_coordinates = get_reference_coordinates_block().into();
                            perturbed_coordinates[node][i] += 0.5 * EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -= EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    let block_1 = get_block();
                    let block_2 = get_block_transformed();
                    assert_eq_within_tols(
                        &block_1
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?,
                        &block_2.helmholtz_free_energy(
                            &get_reference_coordinates_transformed_block().into(),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_block()
                            .helmholtz_free_energy(&get_reference_coordinates_block().into())?
                            .abs(),
                        &0.0,
                    )
                }
            }
        }
        #[test]
        fn nodal_stiffnesses_deformed_symmetry() -> Result<(), TestError> {
            let nodal_stiffness = get_nodal_stiffnesses(true, false)?;
            let result =
                nodal_stiffness
                    .iter()
                    .enumerate()
                    .try_for_each(|(a, nodal_stiffness_a)| {
                        nodal_stiffness_a.iter().enumerate().try_for_each(
                            |(b, nodal_stiffness_ab)| {
                                nodal_stiffness_ab
                                    .iter()
                                    .enumerate()
                                    .zip(nodal_stiffness_ab.transpose().iter())
                                    .try_for_each(
                                        |((i, nodal_stiffness_ab_i), nodal_stiffness_ab_j)| {
                                            nodal_stiffness_ab_i
                                                .iter()
                                                .enumerate()
                                                .zip(nodal_stiffness_ab_j.iter())
                                                .try_for_each(
                                                    |(
                                                        (j, nodal_stiffness_ab_ij),
                                                        nodal_stiffness_ab_ji,
                                                    )| {
                                                        if a == b {
                                                            assert_eq_within_tols(
                                                                nodal_stiffness_ab_ij,
                                                                &nodal_stiffness_ab_ji,
                                                            )
                                                        } else if i == j {
                                                            assert_eq_within_tols(
                                                                nodal_stiffness_ab_ij,
                                                                &nodal_stiffness[b][a][i][j],
                                                            )
                                                        } else {
                                                            Ok(())
                                                        }
                                                    },
                                                )
                                        },
                                    )
                            },
                        )
                    });
            result
        }
        #[test]
        fn nodal_stiffnesses_undeformed_symmetry() -> Result<(), TestError> {
            let nodal_stiffness = get_nodal_stiffnesses(false, false)?;
            let result =
                nodal_stiffness
                    .iter()
                    .enumerate()
                    .try_for_each(|(a, nodal_stiffness_a)| {
                        nodal_stiffness_a.iter().enumerate().try_for_each(
                            |(b, nodal_stiffness_ab)| {
                                nodal_stiffness_ab
                                    .iter()
                                    .enumerate()
                                    .zip(nodal_stiffness_ab.transpose().iter())
                                    .try_for_each(
                                        |((i, nodal_stiffness_ab_i), nodal_stiffness_ab_j)| {
                                            nodal_stiffness_ab_i
                                                .iter()
                                                .enumerate()
                                                .zip(nodal_stiffness_ab_j.iter())
                                                .try_for_each(
                                                    |(
                                                        (j, nodal_stiffness_ab_ij),
                                                        nodal_stiffness_ab_ji,
                                                    )| {
                                                        if a == b {
                                                            assert_eq_within_tols(
                                                                nodal_stiffness_ab_ij,
                                                                &nodal_stiffness_ab_ji,
                                                            )
                                                        } else if i == j {
                                                            assert_eq_within_tols(
                                                                nodal_stiffness_ab_ij,
                                                                &nodal_stiffness[b][a][i][j],
                                                            )
                                                        } else {
                                                            Ok(())
                                                        }
                                                    },
                                                )
                                        },
                                    )
                            },
                        )
                    });
            result
        }
    };
}
pub(crate) use test_helmholtz_free_energy;

macro_rules! test_finite_element_block_with_elastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<NodalStiffnessesBlock, TestError> {
            let block = get_block();
            let mut finite_difference = 0.0;
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
                                            nodal_coordinates[node_b][j] += 0.5 * EPSILON;
                                            finite_difference =
                                                block.nodal_forces(&nodal_coordinates)?[node_a][i];
                                            nodal_coordinates = if is_deformed {
                                                get_coordinates_block()
                                            } else {
                                                get_reference_coordinates_block().into()
                                            };
                                            nodal_coordinates[node_b][j] -= 0.5 * EPSILON;
                                            finite_difference -=
                                                block.nodal_forces(&nodal_coordinates)?[node_a][i];
                                            Ok(finite_difference / EPSILON)
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
        ) -> Result<NodalForcesBlock, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed()
                            .nodal_forces(&get_coordinates_transformed_block())?)
                } else {
                    let converted: TensorRank2<3, 1, 1> =
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
        ) -> Result<NodalStiffnessesBlock, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed()
                            .nodal_stiffnesses(&get_coordinates_transformed_block())?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, 1, 1> =
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
        crate::fem::block::test::test_nodal_forces_and_nodal_stiffnesses!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
    };
}
pub(crate) use test_finite_element_block_with_elastic_constitutive_model;

macro_rules! test_finite_element_block_with_hyperelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        crate::fem::block::test::test_finite_element_block_with_elastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        crate::fem::block::test::test_helmholtz_free_energy!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        use crate::constitutive::solid::AppliedLoad;
        //
        // test in elastic too
        // ok to double-test "solve" of hyperelastic since first one will use root, second one will use minimization
        //
        #[test]
        fn minimize() -> Result<(), TestError> {
            if TEST_SOLVE {
                let dx = 0.88;
                let block = get_block();
                let solution = block.minimize(
                    get_reference_coordinates_block().into(),
                    NewtonRaphson {
                        ..Default::default()
                    },
                )?;
                let (deformation_gradient, _) =
                    $constitutive_model::new($constitutive_model_parameters)
                        .solve(AppliedLoad::UniaxialStress(1.0 + dx))?;
                block.deformation_gradients(&solution).iter().try_for_each(
                    |deformation_gradients| {
                        deformation_gradients
                            .iter()
                            .try_for_each(|deformation_gradient_g| {
                                assert_eq_within_tols(deformation_gradient_g, &deformation_gradient)
                            })
                    },
                )
            } else {
                Ok(())
            }
        }
    };
}
pub(crate) use test_finite_element_block_with_hyperelastic_constitutive_model;

macro_rules! test_finite_element_block_with_viscoelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        fn get_velocities_transformed_block() -> NodalCoordinatesBlock {
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
        ) -> Result<NodalForcesBlock, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed().nodal_forces(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?)
                } else {
                    let converted: TensorRank2<3, 1, 1> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed().nodal_forces(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?)
                }
            } else {
                if is_deformed {
                    Ok(get_block()
                        .nodal_forces(&get_coordinates_block(), &get_velocities_block())?)
                } else {
                    Ok(get_block().nodal_forces(
                        &get_reference_coordinates_block().into(),
                        &NodalVelocitiesBlock::zero(D),
                    )?)
                }
            }
        }
        fn get_nodal_stiffnesses(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<NodalStiffnessesBlock, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_block_transformed().nodal_stiffnesses(
                            &get_coordinates_transformed_block(),
                            &get_velocities_transformed_block(),
                        )?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, 1, 1> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_block_transformed().nodal_stiffnesses(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocitiesBlock::zero(D),
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
                        &NodalVelocitiesBlock::zero(D),
                    )?)
                }
            }
        }
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<NodalStiffnessesBlock, TestError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = 0.0;
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
                                                NodalVelocitiesBlock::zero(D)
                                            };
                                            nodal_velocities[node_a][i] += 0.5 * EPSILON;
                                            finite_difference = block.nodal_forces(
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[node_b][j];
                                            nodal_velocities = if is_deformed {
                                                get_velocities_block()
                                            } else {
                                                NodalVelocitiesBlock::zero(D)
                                            };
                                            nodal_velocities[node_a][i] -= 0.5 * EPSILON;
                                            finite_difference -= block.nodal_forces(
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[node_b][j];
                                            Ok(finite_difference / EPSILON)
                                        })
                                        .collect()
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        }
        crate::fem::block::test::test_nodal_forces_and_nodal_stiffnesses!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
    };
}
pub(crate) use test_finite_element_block_with_viscoelastic_constitutive_model;

macro_rules! test_finite_element_block_with_elastic_hyperviscous_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        crate::fem::block::test::test_finite_element_block_with_viscoelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_parameters
        );
        fn get_finite_difference_of_viscous_dissipation(
            is_deformed: bool,
        ) -> Result<NodalForcesBlock, TestError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = 0.0;
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocitiesBlock::zero(D)
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference =
                                block.viscous_dissipation(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocitiesBlock::zero(D)
                            };
                            nodal_velocities[node][i] -= 0.5 * EPSILON;
                            finite_difference -=
                                block.viscous_dissipation(&nodal_coordinates, &nodal_velocities)?;
                            Ok(finite_difference / EPSILON)
                        })
                        .collect()
                })
                .collect()
        }
        fn get_finite_difference_of_dissipation_potential(
            is_deformed: bool,
        ) -> Result<NodalForcesBlock, TestError> {
            let block = get_block();
            let nodal_coordinates = if is_deformed {
                get_coordinates_block()
            } else {
                get_reference_coordinates_block().into()
            };
            let mut finite_difference = 0.0;
            (0..D)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocitiesBlock::zero(D)
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference = block
                                .dissipation_potential(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocitiesBlock::zero(D)
                            };
                            nodal_velocities[node][i] -= 0.5 * EPSILON;
                            finite_difference -= block
                                .dissipation_potential(&nodal_coordinates, &nodal_velocities)?;
                            Ok(finite_difference / EPSILON)
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
                fn finite_difference() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_forces_0 =
                        block.nodal_forces(&nodal_coordinates, &NodalVelocitiesBlock::zero(D))?;
                    assert_eq_from_fd(
                        &(block.nodal_forces(&nodal_coordinates, &get_velocities_block())?
                            - nodal_forces_0),
                        &get_finite_difference_of_viscous_dissipation(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_velocities = get_velocities_block();
                    let nodal_forces_0 =
                        block.nodal_forces(&nodal_coordinates, &NodalVelocitiesBlock::zero(D))?;
                    let nodal_forces =
                        block.nodal_forces(&nodal_coordinates, &nodal_velocities)? - nodal_forces_0;
                    let minimum = block
                        .viscous_dissipation(&nodal_coordinates, &nodal_velocities)?
                        - nodal_forces.dot(&nodal_velocities);
                    let mut perturbed_velocities = get_velocities_block();
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = get_velocities_block();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.dot(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.dot(&perturbed_velocities)
                                    >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
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
                fn positive() -> Result<(), TestError> {
                    assert!(
                        get_block().viscous_dissipation(
                            &get_coordinates_block(),
                            &get_velocities_block(),
                        )? > 0.0
                    );
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_viscous_dissipation(false)?,
                        &NodalForcesBlock::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum = block
                        .viscous_dissipation(&nodal_coordinates, &NodalVelocitiesBlock::zero(D))?;
                    let mut perturbed_velocities = NodalVelocitiesBlock::zero(D);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = NodalVelocitiesBlock::zero(D);
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
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
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_block().viscous_dissipation(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                        &get_block_transformed().viscous_dissipation(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &get_block().viscous_dissipation(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                        &0.0,
                    )
                }
            }
        }
        mod dissipation_potential {
            use super::*;
            mod deformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_block()
                            .nodal_forces(&get_coordinates_block(), &get_velocities_block())?,
                        &get_finite_difference_of_dissipation_potential(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_coordinates_block();
                    let nodal_velocities = get_coordinates_block();
                    let nodal_forces = block.nodal_forces(&nodal_coordinates, &nodal_velocities)?;
                    let minimum = block
                        .dissipation_potential(&nodal_coordinates, &nodal_velocities)?
                        - nodal_forces.dot(&nodal_velocities);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = nodal_velocities.clone();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.dot(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.dot(&perturbed_velocities)
                                    >= minimum
                            );
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
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
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_dissipation_potential(false)?,
                        &NodalForcesBlock::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum = block.dissipation_potential(
                        &nodal_coordinates,
                        &NodalVelocitiesBlock::zero(D),
                    )?;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = NodalVelocitiesBlock::zero(D);
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
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
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_block().dissipation_potential(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                        &get_block_transformed().dissipation_potential(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &get_block().dissipation_potential(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocitiesBlock::zero(D),
                        )?,
                        &0.0,
                    )
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_with_elastic_hyperviscous_constitutive_model;

macro_rules! test_finite_element_block_with_hyperviscoelastic_constitutive_model
{
    ($block: ident, $element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) =>
    {
        crate::fem::block::test::test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
            $block, $element, $constitutive_model, $constitutive_model_parameters
        );
        // crate::fem::block::test::test_helmholtz_free_energy!($block, $element, $constitutive_model, $constitutive_model_parameters);
        #[test]
        fn dissipation_potential_deformed_positive() -> Result<(), TestError>
        {
            assert!(
                get_block().dissipation_potential(
                    &get_coordinates_block(),
                    &get_velocities_block()
                )? > 0.0
            );
            Ok(())
        }
    }
}
pub(crate) use test_finite_element_block_with_hyperviscoelastic_constitutive_model;
