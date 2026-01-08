macro_rules! test_finite_element_block {
    ($element: ident) => {
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type, $element, G, M, N, P> {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        get_reference_coordinates_block(),
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        get_reference_coordinates_transformed_block(),
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
                        get_reference_coordinates_block(),
                        THICKNESS,
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        get_reference_coordinates_transformed_block(),
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
                use crate::{
                    constitutive::solid::elastic::{
                        AlmansiHamel, SaintVenantKirchhoff,
                        test::{BULK_MODULUS, SHEAR_MODULUS},
                    },
                    fem::block::solid::{
                        SolidFiniteElementBlock, elastic::ElasticFiniteElementBlock,
                    },
                };
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_block_with_elastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        AlmansiHamel
                    );
                }
                // mod hencky {
                //     use super::*;
                //     test_finite_element_block_with_elastic_constitutive_model!(
                //         ElementBlock,
                //         $element,
                //         HenckyType,
                //         HENCKYPARAMETERS
                //     );
                // }
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
                use crate::{
                    constitutive::solid::hyperelastic::{
                        ArrudaBoyce, Fung, Gent, MooneyRivlin, NeoHookean, SaintVenantKirchhoff,
                        Yeoh,
                        test::{
                            EXPONENT, EXTENSIBILITY, EXTRA_MODULUS, NUM_YEOH_EXTRA_MODULI,
                            NUMBER_OF_LINKS, YEOH_EXTRA_MODULI,
                        },
                    },
                    fem::block::solid::{
                        SolidFiniteElementBlock, elastic::ElasticFiniteElementBlock,
                        hyperelastic::HyperelasticFiniteElementBlock,
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
                // mod hencky {
                //     use super::*;
                //     test_finite_element_block_with_hyperelastic_constitutive_model!(
                //         ElementBlock,
                //         $element,
                //         HenckyType,
                //         HENCKYPARAMETERS
                //     );
                // }
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
                    type YeohType = Yeoh<NUM_YEOH_EXTRA_MODULI>;
                    test_finite_element_block_with_hyperelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        Yeoh {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extra_moduli: YEOH_EXTRA_MODULI,
                        },
                        YeohType
                    );
                }
            }
            mod elastic_hyperviscous {
                use super::*;
                use crate::{
                    constitutive::solid::elastic_hyperviscous::{
                        AlmansiHamel,
                        test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                    },
                    fem::block::solid::{
                        SolidFiniteElementBlock,
                        elastic_hyperviscous::ElasticHyperviscousFiniteElementBlock,
                        viscoelastic::ViscoelasticFiniteElementBlock,
                    },
                };
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            bulk_viscosity: BULK_VISCOSITY,
                            shear_viscosity: SHEAR_VISCOSITY,
                        },
                        AlmansiHamel
                    );
                }
            }
            mod hyperviscoelastic {
                use super::*;
                use crate::{
                    constitutive::solid::hyperviscoelastic::{
                        SaintVenantKirchhoff,
                        test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                    },
                    fem::block::solid::{
                        SolidFiniteElementBlock,
                        elastic_hyperviscous::ElasticHyperviscousFiniteElementBlock,
                        viscoelastic::ViscoelasticFiniteElementBlock,
                    },
                };
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoff {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            bulk_viscosity: BULK_VISCOSITY,
                            shear_viscosity: SHEAR_VISCOSITY,
                        },
                        SaintVenantKirchhoff
                    );
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_inner;

macro_rules! test_nodal_forces_and_nodal_stiffnesses {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        setup_block!($constitutive_model, $constitutive_model_type);
        fn get_coordinates_transformed_block() -> NodalCoordinates {
            get_coordinates_block()
                .iter()
                .map(|coordinate| {
                    (get_rotation_current_configuration() * coordinate)
                        + get_translation_current_configuration()
                })
                .collect()
        }
        fn get_reference_coordinates_transformed_block() -> NodalReferenceCoordinates {
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
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(
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
        //             + get_connectivity().len() * std::mem::size_of::<$element::<$constitutive_model_type>>()
        //     )
        // }
    };
}
pub(crate) use test_nodal_forces_and_nodal_stiffnesses;

macro_rules! test_helmholtz_free_energy {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_finite_difference_of_helmholtz_free_energy(
            is_deformed: bool,
        ) -> Result<NodalForcesSolid, TestError> {
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
                        - nodal_forces.full_contraction(&nodal_coordinates);
                    let mut perturbed = 0.0;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_coordinates = nodal_coordinates.clone();
                            perturbed_coordinates[node][i] += 0.5 * EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - nodal_forces.full_contraction(&perturbed_coordinates);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -= EPSILON;
                            perturbed = block.helmholtz_free_energy(&perturbed_coordinates)?
                                - nodal_forces.full_contraction(&perturbed_coordinates);
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
                        &NodalForcesSolid::zero(D),
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
                                nodal_stiffness_ab.iter().enumerate().try_for_each(
                                    |(i, nodal_stiffness_ab_i)| {
                                        nodal_stiffness_ab_i.iter().enumerate().try_for_each(
                                            |(j, nodal_stiffness_ab_ij)| {
                                                assert_eq_within_tols(
                                                    nodal_stiffness_ab_ij,
                                                    &nodal_stiffness[b][a][j][i],
                                                )
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
                                nodal_stiffness_ab.iter().enumerate().try_for_each(
                                    |(i, nodal_stiffness_ab_i)| {
                                        nodal_stiffness_ab_i.iter().enumerate().try_for_each(
                                            |(j, nodal_stiffness_ab_ij)| {
                                                assert_eq_within_tols(
                                                    nodal_stiffness_ab_ij,
                                                    &nodal_stiffness[b][a][j][i],
                                                )
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

macro_rules! test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<NodalStiffnessesSolid, TestError> {
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
        ) -> Result<NodalForcesSolid, TestError> {
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
        ) -> Result<NodalStiffnessesSolid, TestError> {
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
            $constitutive_model_type
        );
        macro_rules! test_root_with_solver {
            ($solver: ident) => {
                #[test]
                fn root() -> Result<(), TestError> {
                    let (applied_load, a, b) = equality_constraint();
                    let block = get_block();
                    let coordinates = FirstOrderRoot::root(
                        &block,
                        EqualityConstraint::Linear(a, b),
                        $solver::default(),
                    )?;
                    let deformation_gradient =
                        $constitutive_model.root(applied_load, $solver::default())?;
                    block
                        .deformation_gradients(&coordinates)
                        .iter()
                        .try_for_each(|deformation_gradients| {
                            deformation_gradients
                                .iter()
                                .try_for_each(|deformation_gradient_g| {
                                    assert_eq_within_tols(
                                        deformation_gradient_g,
                                        &deformation_gradient,
                                    )
                                })
                        })
                }
            };
        }
        mod newton_raphson_root {
            use super::*;
            use crate::{
                constitutive::solid::elastic::FirstOrderRoot as _, fem::block::FirstOrderRoot,
                math::optimize::NewtonRaphson,
            };
            test_root_with_solver!(NewtonRaphson);
        }
    };
}
pub(crate) use test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model;

macro_rules! test_finite_element_block_with_elastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::test::test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        #[test]
        fn nodal_stiffnesses_deformed_non_symmetry() -> Result<(), TestError> {
            let nodal_stiffness = get_nodal_stiffnesses(true, false)?;
            let n = nodal_stiffness.len();
            assert!(
                assert_eq_within_tols(
                    &nodal_stiffness,
                    &(0..n)
                        .map(|a| (0..n)
                            .map(|b| (0..3)
                                .map(|i| (0..3)
                                    .map(|j| nodal_stiffness[b][a][j][i].clone())
                                    .collect())
                                .collect())
                            .collect())
                        .collect()
                )
                .is_err()
            );
            Ok(())
        }
    };
}
pub(crate) use test_finite_element_block_with_elastic_constitutive_model;

macro_rules! test_finite_element_block_with_hyperelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::test::test_finite_element_block_with_elastic_or_hyperelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        crate::fem::block::test::test_helmholtz_free_energy!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        macro_rules! test_minimize_with_solver {
            ($solver: ident) => {
                #[test]
                fn minimize() -> Result<(), TestError> {
                    let (applied_load, a, b) = equality_constraint();
                    let block = get_block();
                    let coordinates =
                        SecondOrderMinimize::minimize(&block, EqualityConstraint::Linear(a, b), $solver::default())?;
                    let deformation_gradient =
                        $constitutive_model
                            .minimize(applied_load, $solver::default())?;
                    block
                        .deformation_gradients(&coordinates)
                        .iter()
                        .try_for_each(|deformation_gradients| {
                            deformation_gradients
                                .iter()
                                .try_for_each(|deformation_gradient_g| {
                                    assert_eq_within_tols(
                                        deformation_gradient_g,
                                        &deformation_gradient,
                                    )
                                })
                        })
                }
            };
        }
        mod newton_raphson_minimize {
            use super::*;
            use crate::{
                constitutive::solid::hyperelastic::SecondOrderMinimize as _,
                fem::block::SecondOrderMinimize, math::optimize::NewtonRaphson,
            };
            test_minimize_with_solver!(NewtonRaphson);
        }
    };
}
pub(crate) use test_finite_element_block_with_hyperelastic_constitutive_model;

macro_rules! test_finite_element_block_with_viscoelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_velocities_transformed_block() -> NodalCoordinates {
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
        ) -> Result<NodalForcesSolid, TestError> {
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
        ) -> Result<NodalStiffnessesSolid, TestError> {
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
        ) -> Result<NodalStiffnessesSolid, TestError> {
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
                                                NodalVelocities::zero(D)
                                            };
                                            nodal_velocities[node_a][i] += 0.5 * EPSILON;
                                            finite_difference = block.nodal_forces(
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[node_b][j];
                                            nodal_velocities = if is_deformed {
                                                get_velocities_block()
                                            } else {
                                                NodalVelocities::zero(D)
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
            integrate::{BogackiShampine, DormandPrince, Verner8, Verner9},
            optimize::NewtonRaphson,
        };
        macro_rules! test_with_integrator {
            ($integrator: ident) => {
                #[test]
                fn minimize() -> Result<(), TestError> {
                    use crate::constitutive::solid::elastic_hyperviscous::SecondOrderMinimize as _;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) = block.minimize(
                        EqualityConstraint::Linear(a, b),
                        $integrator::default(),
                        &[0.0, 1.0],
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
                                                assert_eq_within_tols(
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
                                                assert_eq_within_tols(
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
                fn root() -> Result<(), TestError> {
                    use crate::constitutive::solid::viscoelastic::FirstOrderRoot as _;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) = block.root(
                        EqualityConstraint::Linear(a, b),
                        $integrator::default(),
                        &[0.0, 1.0],
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
                                                assert_eq_within_tols(
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
                                                assert_eq_within_tols(
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
        ) -> Result<NodalForcesSolid, TestError> {
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
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference =
                                block.viscous_dissipation(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
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
        ) -> Result<NodalForcesSolid, TestError> {
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
                                NodalVelocities::zero(D)
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference = block
                                .dissipation_potential(&nodal_coordinates, &nodal_velocities)?;
                            nodal_velocities = if is_deformed {
                                get_velocities_block()
                            } else {
                                NodalVelocities::zero(D)
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
                        block.nodal_forces(&nodal_coordinates, &NodalVelocities::zero(D))?;
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
                        block.nodal_forces(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    let nodal_forces =
                        block.nodal_forces(&nodal_coordinates, &nodal_velocities)? - nodal_forces_0;
                    let minimum = block
                        .viscous_dissipation(&nodal_coordinates, &nodal_velocities)?
                        - nodal_forces.full_contraction(&nodal_velocities);
                    let mut perturbed_velocities = get_velocities_block();
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = get_velocities_block();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.full_contraction(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
                            assert!(
                                block.viscous_dissipation(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.full_contraction(&perturbed_velocities)
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
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum =
                        block.viscous_dissipation(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    let mut perturbed_velocities = NodalVelocities::zero(D);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = NodalVelocities::zero(D);
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
                            &NodalVelocities::zero(D),
                        )?,
                        &get_block_transformed().viscous_dissipation(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &get_block().viscous_dissipation(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocities::zero(D),
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
                        - nodal_forces.full_contraction(&nodal_velocities);
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = nodal_velocities.clone();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.full_contraction(&perturbed_velocities)
                                    >= minimum
                            );
                            perturbed_velocities[node][i] -= EPSILON;
                            assert!(
                                block.dissipation_potential(
                                    &nodal_coordinates,
                                    &perturbed_velocities,
                                )? - nodal_forces.full_contraction(&perturbed_velocities)
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
                        &NodalForcesSolid::zero(D),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let block = get_block();
                    let nodal_coordinates = get_reference_coordinates_block().into();
                    let minimum = block
                        .dissipation_potential(&nodal_coordinates, &NodalVelocities::zero(D))?;
                    (0..D).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            let mut perturbed_velocities = NodalVelocities::zero(D);
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
                            &NodalVelocities::zero(D),
                        )?,
                        &get_block_transformed().dissipation_potential(
                            &get_reference_coordinates_transformed_block().into(),
                            &NodalVelocities::zero(D),
                        )?,
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(
                        &get_block().dissipation_potential(
                            &get_reference_coordinates_block().into(),
                            &NodalVelocities::zero(D),
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
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) =>
    {
        crate::fem::block::test::test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
            $block, $element, $constitutive_model, $constitutive_model_type
        );
        // crate::fem::block::test::test_helmholtz_free_energy!($block, $element, $constitutive_model, $constitutive_model_type);
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
