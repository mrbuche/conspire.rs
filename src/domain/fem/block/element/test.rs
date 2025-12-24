use crate::mechanics::Scalar;
pub const THICKNESS: Scalar = 1.23;

macro_rules! test_finite_element {
    ($element: ident) => {
        use crate::mechanics::test::{get_deformation_gradient, get_deformation_gradient_rate};
        crate::fem::block::element::test::setup!();
        fn coordinates() -> ElementNodalCoordinates<N> {
            get_deformation_gradient() * reference_coordinates()
        }
        fn velocities() -> ElementNodalVelocities<N> {
            get_deformation_gradient_rate() * reference_coordinates()
        }
        fn element() -> $element {
            $element::from(reference_coordinates())
        }
        fn element_transformed() -> $element {
            $element::from(reference_coordinates_transformed())
        }
        #[test]
        fn size() {
            assert_eq!(
                std::mem::size_of::<$element>(),
                std::mem::size_of::<GradientVectors<G, N>>() + std::mem::size_of::<ScalarList<G>>()
            )
        }
        macro_rules! setup_element {
            () => {
                fn get_element() -> $element {
                    $element::from(reference_coordinates())
                }
                fn get_element_transformed() -> $element {
                    $element::from(reference_coordinates_transformed())
                }
            };
        }
        crate::fem::block::element::test::test_finite_element_inner!($element);
    };
}
pub(crate) use test_finite_element;

macro_rules! test_surface_finite_element {
    ($element: ident) => {
        use crate::{
            fem::block::element::test::setup,
            math::{Rank2, TensorArray},
            mechanics::RotationCurrentConfiguration,
        };
        fn get_deformation_gradient_special() -> DeformationGradient {
            DeformationGradient::from([[0.62, 0.20, 0.00], [0.32, 0.98, 0.00], [0.00, 0.00, 1.00]])
        }
        fn get_deformation_gradient_rate_special() -> DeformationGradientRate {
            DeformationGradient::from([[0.53, 0.58, 0.00], [0.28, 0.77, 0.00], [0.00, 0.00, 0.00]])
        }
        fn get_deformation_gradient() -> DeformationGradient {
            get_deformation_gradient_rotation() * get_deformation_gradient_special()
        }
        fn get_deformation_gradient_rate() -> DeformationGradientRate {
            get_deformation_gradient_rotation() * get_deformation_gradient_rate_special()
        }
        fn get_deformation_gradient_rotation() -> RotationCurrentConfiguration {
            get_rotation_reference_configuration().transpose().into()
        }
        setup!();
        fn coordinates() -> ElementNodalCoordinates<N> {
            get_deformation_gradient() * reference_coordinates()
        }
        fn velocities() -> ElementNodalVelocities<N> {
            get_deformation_gradient_rate() * reference_coordinates()
        }
        fn element() -> $element {
            $element::from((reference_coordinates(), THICKNESS))
        }
        fn element_transformed() -> $element {
            $element::from((reference_coordinates_transformed(), THICKNESS))
        }
        #[test]
        fn size() {
            assert_eq!(
                std::mem::size_of::<$element>(),
                std::mem::size_of::<GradientVectors<G, N>>()
                    + std::mem::size_of::<ScalarList<G>>()
                    + std::mem::size_of::<Normals<G>>()
            )
        }
        macro_rules! setup_element {
            () => {
                fn get_element() -> $element {
                    $element::from((reference_coordinates(), THICKNESS))
                }
                fn get_element_transformed() -> $element {
                    $element::from((reference_coordinates_transformed(), THICKNESS))
                }
            };
        }
        crate::fem::block::element::test::test_finite_element_inner!($element);
        use crate::{
            EPSILON,
            math::test::{TestError, assert_eq_from_fd, assert_eq_within_tols},
        };
        mod bases {
            use super::*;
            #[test]
            fn objectivity() -> Result<(), TestError> {
                $element::bases(&coordinates_transformed())
                    .iter()
                    .zip($element::bases(&coordinates()).iter())
                    .try_for_each(|(basis_transformed, basis)| {
                        basis_transformed.iter().zip(basis.iter()).try_for_each(
                            |(basis_transformed_m, basis_m)| {
                                assert_eq_within_tols(
                                    &(get_rotation_current_configuration().transpose()
                                        * basis_transformed_m),
                                    basis_m,
                                )
                            },
                        )
                    })
            }
        }
        mod dual_bases {
            #[test]
            fn basis() -> Result<(), TestError> {
                let mut surface_identity = DeformationGradient::identity();
                surface_identity[2][2] = 0.0;
                $element::bases(&coordinates())
                    .iter()
                    .zip($element::dual_bases(&coordinates()).iter())
                    .try_for_each(|(basis, dual_basis)| {
                        assert_eq_within_tols(
                            &basis
                                .iter()
                                .map(|basis_m| {
                                    dual_basis
                                        .iter()
                                        .map(|dual_basis_n| basis_m * dual_basis_n)
                                        .collect()
                                })
                                .collect(),
                            &surface_identity,
                        )
                    })
            }
            use super::*;
            #[test]
            fn objectivity() -> Result<(), TestError> {
                $element::dual_bases(&coordinates_transformed())
                    .iter()
                    .zip($element::dual_bases(&coordinates()).iter())
                    .try_for_each(|(basis_transformed, basis)| {
                        basis_transformed.iter().zip(basis.iter()).try_for_each(
                            |(basis_transformed_m, basis_m)| {
                                assert_eq_within_tols(
                                    &(get_rotation_current_configuration().transpose()
                                        * basis_transformed_m),
                                    basis_m,
                                )
                            },
                        )
                    })
            }
        }
        mod normals {
            use super::*;
            #[test]
            fn finite_difference() -> Result<(), TestError> {
                let mut finite_difference = 0.0;
                let normal_gradients_from_fd = (0..G)
                    .map(|p| {
                        (0..N)
                            .map(|a| {
                                (0..3)
                                    .map(|m| {
                                        (0..3)
                                            .map(|i| {
                                                let mut nodal_coordinates = coordinates();
                                                nodal_coordinates[a][m] += 0.5 * EPSILON;
                                                finite_difference =
                                                    $element::normals(&nodal_coordinates)[p][i];
                                                nodal_coordinates[a][m] -= EPSILON;
                                                finite_difference -=
                                                    $element::normals(&nodal_coordinates)[p][i];
                                                finite_difference / EPSILON
                                            })
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                assert_eq_from_fd(
                    &$element::normal_gradients(&coordinates()),
                    &normal_gradients_from_fd,
                )
            }
            #[test]
            fn normal() -> Result<(), TestError> {
                $element::bases(&coordinates())
                    .iter()
                    .zip(
                        $element::dual_bases(&coordinates())
                            .iter()
                            .zip($element::normals(&coordinates()).iter()),
                    )
                    .try_for_each(|(basis, (dual_basis, normal))| {
                        assert_eq_within_tols(&(&basis[0] * normal), &0.0)?;
                        assert_eq_within_tols(&(&basis[1] * normal), &0.0)?;
                        assert_eq_within_tols(&(&dual_basis[0] * normal), &0.0)?;
                        assert_eq_within_tols(&(&dual_basis[1] * normal), &0.0)
                    })
            }
            #[test]
            fn normalized() -> Result<(), TestError> {
                $element::normals(&coordinates())
                    .iter()
                    .try_for_each(|normal| assert_eq_within_tols(&normal.norm(), &1.0))
            }
            #[test]
            fn objectivity() -> Result<(), TestError> {
                $element::normals(&coordinates_transformed())
                    .iter()
                    .zip($element::normals(&coordinates()).iter())
                    .try_for_each(|(normal_transformed, normal)| {
                        assert_eq_within_tols(
                            &(get_rotation_current_configuration().transpose()
                                * normal_transformed),
                            normal,
                        )
                    })
            }
        }
        mod normal_gradients {
            use super::*;
            #[test]
            fn objectivity() -> Result<(), TestError> {
                $element::normal_gradients(&coordinates_transformed())
                    .iter()
                    .zip($element::normal_gradients(&coordinates()).iter())
                    .try_for_each(|(normal_gradient_transformed, normal_gradient)| {
                        normal_gradient_transformed
                            .iter()
                            .zip(normal_gradient.iter())
                            .try_for_each(|(normal_gradient_transformed_a, normal_gradient_a)| {
                                assert_eq_within_tols(
                                    &(get_rotation_current_configuration().transpose()
                                        * normal_gradient_transformed_a
                                        * get_rotation_current_configuration()),
                                    normal_gradient_a,
                                )
                            })
                    })
            }
        }
        mod normal_rate {
            use super::*;
            #[test]
            fn finite_difference() -> Result<(), TestError> {
                let mut finite_difference = 0.0;
                let normal_rates_from_fd = (0..G)
                    .map(|p| {
                        (0..3)
                            .map(|i| {
                                velocities()
                                    .iter()
                                    .enumerate()
                                    .map(|(a, velocity_a)| {
                                        velocity_a
                                            .iter()
                                            .enumerate()
                                            .map(|(k, velocity_a_k)| {
                                                let mut nodal_coordinates = coordinates();
                                                nodal_coordinates[a][k] += 0.5 * EPSILON;
                                                finite_difference =
                                                    $element::normals(&nodal_coordinates)[p][i];
                                                nodal_coordinates[a][k] -= EPSILON;
                                                finite_difference -=
                                                    $element::normals(&nodal_coordinates)[p][i];
                                                finite_difference / EPSILON * velocity_a_k
                                            })
                                            .sum::<Scalar>()
                                    })
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                assert_eq_from_fd(
                    &$element::normal_rates(&coordinates(), &velocities()),
                    &normal_rates_from_fd,
                )
            }
            #[test]
            fn objectivity() -> Result<(), TestError> {
                $element::normals(&coordinates_transformed())
                    .iter()
                    .zip(
                        $element::normal_rates(
                            &coordinates_transformed(),
                            &velocities_transformed(),
                        )
                        .iter()
                        .zip($element::normal_rates(&coordinates(), &velocities()).iter()),
                    )
                    .try_for_each(
                        |(normal_transformed, (normal_rate_transformed, normal_rate))| {
                            assert_eq_within_tols(
                                &(get_rotation_current_configuration().transpose()
                                    * normal_rate_transformed
                                    + get_rotation_rate_current_configuration().transpose()
                                        * normal_transformed),
                                normal_rate,
                            )
                        },
                    )
            }
        }
        mod reference_normals {
            use super::*;
            #[test]
            fn normal() -> Result<(), TestError> {
                $element::bases(&reference_coordinates())
                    .iter()
                    .zip(
                        $element::dual_bases(&reference_coordinates())
                            .iter()
                            .zip(element().reference_normals().iter()),
                    )
                    .try_for_each(|(basis, (dual_basis, reference_normal))| {
                        assert_eq_within_tols(&(&basis[0] * reference_normal), &0.0)?;
                        assert_eq_within_tols(&(&basis[1] * reference_normal), &0.0)?;
                        assert_eq_within_tols(&(&dual_basis[0] * reference_normal), &0.0)?;
                        assert_eq_within_tols(&(&dual_basis[1] * reference_normal), &0.0)
                    })
            }
            #[test]
            fn normalized() -> Result<(), TestError> {
                element()
                    .reference_normals()
                    .iter()
                    .try_for_each(|reference_normal| {
                        assert_eq_within_tols(&reference_normal.norm(), &1.0)
                    })
            }
            #[test]
            fn objectivity() -> Result<(), TestError> {
                element_transformed()
                    .reference_normals()
                    .iter()
                    .zip(element().reference_normals().iter())
                    .try_for_each(|(reference_normal_transformed, reference_normal)| {
                        assert_eq_within_tols(
                            &(get_rotation_reference_configuration().transpose()
                                * reference_normal_transformed),
                            reference_normal,
                        )
                    })
            }
        }
    };
}
pub(crate) use test_surface_finite_element;

macro_rules! setup {
    () => {
        use crate::{
            constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            mechanics::test::{
                get_rotation_current_configuration, get_rotation_rate_current_configuration,
                get_rotation_reference_configuration, get_translation_current_configuration,
                get_translation_rate_current_configuration,
                get_translation_reference_configuration,
            },
        };
        fn coordinates_transformed() -> ElementNodalCoordinates<N> {
            coordinates()
                .iter()
                .map(|coordinate| {
                    get_rotation_current_configuration() * coordinate
                        + get_translation_current_configuration()
                })
                .collect()
        }
        fn reference_coordinates_transformed() -> ElementNodalReferenceCoordinates<N> {
            reference_coordinates()
                .iter()
                .map(|reference_coordinate| {
                    get_rotation_reference_configuration() * reference_coordinate
                        + get_translation_reference_configuration()
                })
                .collect()
        }
        fn velocities_transformed() -> ElementNodalVelocities<N> {
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
    };
}
pub(crate) use setup;

macro_rules! test_finite_element_inner {
    ($element: ident) => {
        mod element {
            use super::*;
            // use super::{
            //     DeformationGradientList, DeformationGradientRateList, ElementNodalVelocities, G,
            //     Rank2, SolidFiniteElement, Tensor, TensorArray, TestError, assert_eq_within_tols,
            //     coordinates, coordinates_transformed, element, element_transformed,
            //     get_deformation_gradient, get_deformation_gradient_rate,
            //     get_rotation_current_configuration, get_rotation_rate_current_configuration,
            //     get_rotation_reference_configuration, reference_coordinates,
            //     reference_coordinates_transformed, velocities, velocities_transformed, $element,
            // };
            use crate::{
                EPSILON,
                fem::block::element::test::{
                    test_finite_element_with_elastic_constitutive_model,
                    test_finite_element_with_elastic_hyperviscous_constitutive_model,
                    test_finite_element_with_hyperelastic_constitutive_model,
                    test_finite_element_with_hyperviscoelastic_constitutive_model,
                },
                math::{
                    Rank2, TensorArray, TensorRank2,
                    test::{TestError, assert_eq, assert_eq_from_fd, assert_eq_within_tols},
                },
                mechanics::{
                    Scalar,
                    test::{
                        get_rotation_current_configuration,
                        get_rotation_rate_current_configuration,
                        get_rotation_reference_configuration,
                    },
                },
            };
            mod shape_functions {
                use super::*;
                use crate::EPSILON;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    let mut finite_difference = 0.0;
                    $element::integration_points()
                        .into_iter()
                        .zip($element::shape_functions_gradients_at_integration_points())
                        .try_for_each(|(mut integration_point, shape_functions_gradients)| {
                            assert_eq_from_fd(
                                &shape_functions_gradients,
                                &(0..N)
                                    .map(|n| {
                                        (0..M)
                                            .map(|m| {
                                                integration_point[m] += 0.5 * EPSILON;
                                                finite_difference = $element::shape_functions(
                                                    integration_point.clone(),
                                                )[n];
                                                integration_point[m] -= EPSILON;
                                                finite_difference -= $element::shape_functions(
                                                    integration_point.clone(),
                                                )[n];
                                                integration_point[m] += 0.5 * EPSILON;
                                                finite_difference / EPSILON
                                            })
                                            .collect()
                                    })
                                    .collect(),
                            )
                        })
                }
                #[test]
                fn partition_of_unity() -> Result<(), TestError> {
                    $element::shape_functions_at_integration_points()
                        .iter()
                        .try_for_each(|shape_functions| {
                            assert_eq_within_tols(&shape_functions.iter().sum(), &1.0)
                        })
                }
            }
            mod shape_functions_gradients {
                use super::*;
                #[test]
                fn partition_of_unity() -> Result<(), TestError> {
                    let mut sums = [0.0; M];
                    $element::shape_functions_gradients_at_integration_points()
                        .iter()
                        .try_for_each(|shape_functions_gradients| {
                            sums = [0.0; M];
                            shape_functions_gradients.iter().for_each(|row| {
                                row.iter()
                                    .zip(sums.iter_mut())
                                    .for_each(|(entry, sum)| *sum += entry)
                            });
                            sums.iter()
                                .try_for_each(|sum| assert_eq_within_tols(sum, &0.0))
                        })
                }
            }
            mod solid {
                use super::*;
                fn deformation_gradients() -> DeformationGradientList<G> {
                    (0..G).map(|_| get_deformation_gradient()).collect()
                }
                fn deformation_gradient_rates() -> DeformationGradientRateList<G> {
                    (0..G).map(|_| get_deformation_gradient_rate()).collect()
                }
                mod deformation_gradient {
                    use super::*;
                    mod deformed {
                        use super::*;
                        #[test]
                        fn calculate() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element().deformation_gradients(&coordinates()),
                                &deformation_gradients(),
                            )
                        }
                        #[test]
                        fn objectivity() -> Result<(), TestError> {
                            element()
                                .deformation_gradients(&coordinates())
                                .iter()
                                .zip(
                                    element_transformed()
                                        .deformation_gradients(&coordinates_transformed())
                                        .iter(),
                                )
                                .try_for_each(
                                    |(deformation_gradient, deformation_gradient_transformed)| {
                                        assert_eq_within_tols(
                                            deformation_gradient,
                                            &(get_rotation_current_configuration().transpose()
                                                * deformation_gradient_transformed
                                                * get_rotation_reference_configuration()),
                                        )
                                    },
                                )
                        }
                    }
                    mod undeformed {
                        use super::*;
                        #[test]
                        fn calculate() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element().deformation_gradients(&reference_coordinates().into()),
                                &DeformationGradientList::identity(),
                            )
                        }
                        #[test]
                        fn objectivity() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element_transformed().deformation_gradients(
                                    &reference_coordinates_transformed().into(),
                                ),
                                &DeformationGradientList::identity(),
                            )
                        }
                    }
                }
                mod deformation_gradient_rate {
                    use super::*;
                    mod deformed {
                        use super::*;
                        #[test]
                        fn calculate() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element()
                                    .deformation_gradient_rates(&coordinates(), &velocities()),
                                &deformation_gradient_rates(),
                            )
                        }
                        #[test]
                        fn objectivity() -> Result<(), TestError> {
                            element()
                                .deformation_gradients(&coordinates())
                                .iter()
                                .zip(
                                    element()
                                        .deformation_gradient_rates(&coordinates(), &velocities())
                                        .iter()
                                        .zip(
                                            element_transformed()
                                                .deformation_gradient_rates(
                                                    &coordinates_transformed(),
                                                    &velocities_transformed(),
                                                )
                                                .iter(),
                                        ),
                                )
                                .try_for_each(
                                    |(
                                        deformation_gradient,
                                        (
                                            deformation_gradient_rate,
                                            deformation_gradient_rate_transformed,
                                        ),
                                    )| {
                                        assert_eq_within_tols(
                                            deformation_gradient_rate,
                                            &(get_rotation_current_configuration().transpose()
                                                * (deformation_gradient_rate_transformed
                                                    * get_rotation_reference_configuration()
                                                    - get_rotation_rate_current_configuration()
                                                        * deformation_gradient)),
                                        )
                                    },
                                )
                        }
                    }
                    mod undeformed {
                        use super::*;
                        #[test]
                        fn calculate() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element().deformation_gradient_rates(
                                    &reference_coordinates().into(),
                                    &ElementNodalVelocities::zero().into(),
                                ),
                                &DeformationGradientRateList::zero(),
                            )
                        }
                        #[test]
                        fn objectivity() -> Result<(), TestError> {
                            assert_eq_within_tols(
                                &element_transformed().deformation_gradient_rates(
                                    &reference_coordinates_transformed().into(),
                                    &ElementNodalVelocities::zero().into(),
                                ),
                                &DeformationGradientRateList::zero(),
                            )
                        }
                    }
                }
            }
            mod elastic {
                use super::*;
                use crate::{
                    constitutive::solid::elastic::{
                        AlmansiHamel, Hencky, SaintVenantKirchhoff,
                        test::{BULK_MODULUS, SHEAR_MODULUS},
                    },
                    fem::block::element::solid::{
                        ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                        elastic::ElasticFiniteElement,
                    },
                };
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_with_elastic_constitutive_model!(
                        $element,
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        AlmansiHamel
                    );
                }
                mod hencky {
                    use super::*;
                    test_finite_element_with_elastic_constitutive_model!(
                        $element,
                        Hencky {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        Hencky
                    );
                }
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_with_elastic_constitutive_model!(
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
                        ArrudaBoyce, Fung, Gent, Hencky, MooneyRivlin, NeoHookean,
                        SaintVenantKirchhoff, Yeoh,
                        test::{
                            EXPONENT, EXTENSIBILITY, EXTRA_MODULUS, NUMBER_OF_LINKS,
                            YEOH_EXTRA_MODULI,
                        },
                    },
                    fem::block::element::solid::{
                        ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                        elastic::ElasticFiniteElement, hyperelastic::HyperelasticFiniteElement,
                    },
                };
                mod arruda_boyce {
                    use super::*;
                    test_finite_element_with_hyperelastic_constitutive_model!(
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
                    test_finite_element_with_hyperelastic_constitutive_model!(
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
                    test_finite_element_with_hyperelastic_constitutive_model!(
                        $element,
                        Gent {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extensibility: EXTENSIBILITY,
                        },
                        Gent
                    );
                }
                mod hencky {
                    use super::*;
                    test_finite_element_with_hyperelastic_constitutive_model!(
                        $element,
                        Hencky {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        Hencky
                    );
                }
                mod mooney_rivlin {
                    use super::*;
                    test_finite_element_with_hyperelastic_constitutive_model!(
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
                    test_finite_element_with_hyperelastic_constitutive_model!(
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
                    test_finite_element_with_hyperelastic_constitutive_model!(
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
                    test_finite_element_with_hyperelastic_constitutive_model!(
                        $element,
                        Yeoh {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            extra_moduli: YEOH_EXTRA_MODULI,
                        },
                        Yeoh
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
                    fem::block::element::solid::{
                        ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                        elastic_hyperviscous::ElasticHyperviscousFiniteElement,
                        viscoelastic::ViscoelasticFiniteElement,
                    },
                };
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_with_elastic_hyperviscous_constitutive_model!(
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
                    fem::block::element::solid::{
                        ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                        elastic_hyperviscous::ElasticHyperviscousFiniteElement,
                        hyperviscoelastic::HyperviscoelasticFiniteElement,
                        viscoelastic::ViscoelasticFiniteElement,
                    },
                };
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_with_hyperviscoelastic_constitutive_model!(
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
pub(crate) use test_finite_element_inner;

macro_rules! test_nodal_forces_and_nodal_stiffnesses {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        setup_element!();
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
                        &get_nodal_forces(true, false, true)?,
                        &get_nodal_forces(true, true, true)?,
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
                        &get_nodal_forces(false, true, true)?,
                        &ElementNodalForcesSolid::zero(),
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(
                        &get_nodal_forces(false, false, false)?,
                        &ElementNodalForcesSolid::zero(),
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
    };
}
pub(crate) use test_nodal_forces_and_nodal_stiffnesses;

macro_rules! test_helmholtz_free_energy {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_helmholtz_free_energy(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<Scalar, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_element_transformed()
                        .helmholtz_free_energy(&$constitutive_model, &coordinates_transformed())?)
                } else {
                    Ok(get_element_transformed().helmholtz_free_energy(
                        &$constitutive_model,
                        &reference_coordinates_transformed().into(),
                    )?)
                }
            } else {
                if is_deformed {
                    Ok(
                        get_element()
                            .helmholtz_free_energy(&$constitutive_model, &coordinates())?,
                    )
                } else {
                    Ok(get_element().helmholtz_free_energy(
                        &$constitutive_model,
                        &reference_coordinates().into(),
                    )?)
                }
            }
        }
        fn get_finite_difference_of_helmholtz_free_energy(
            is_deformed: bool,
        ) -> Result<ElementNodalForcesSolid<N>, TestError> {
            let element = get_element();
            let mut finite_difference = 0.0;
            (0..N)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let mut nodal_coordinates = if is_deformed {
                                coordinates()
                            } else {
                                reference_coordinates().into()
                            };
                            nodal_coordinates[node][i] += 0.5 * EPSILON;
                            finite_difference = element
                                .helmholtz_free_energy(&$constitutive_model, &nodal_coordinates)?;
                            nodal_coordinates[node][i] -= EPSILON;
                            finite_difference -= element
                                .helmholtz_free_energy(&$constitutive_model, &nodal_coordinates)?;
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
                        &get_nodal_forces(true, false, false)?,
                        &get_finite_difference_of_helmholtz_free_energy(true)?,
                    )
                }
                #[test]
                #[should_panic(expected = "Invalid Jacobian")]
                fn invalid_jacobian() {
                    let mut deformation_gradient = DeformationGradient::identity();
                    deformation_gradient[0][0] = 0.0;
                    get_element()
                        .helmholtz_free_energy(
                            &$constitutive_model,
                            &(deformation_gradient * reference_coordinates()),
                        )
                        .unwrap();
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let nodal_forces = get_nodal_forces(true, false, false)?;
                    let minimum = get_helmholtz_free_energy(true, false)?
                        - nodal_forces.full_contraction(&coordinates());
                    let mut perturbed = 0.0;
                    let mut perturbed_coordinates = coordinates();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_coordinates = coordinates();
                            perturbed_coordinates[node][i] += 0.5 * EPSILON;
                            perturbed = element.helmholtz_free_energy(
                                &$constitutive_model,
                                &perturbed_coordinates,
                            )? - nodal_forces.full_contraction(&perturbed_coordinates);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -= EPSILON;
                            perturbed = element.helmholtz_free_energy(
                                &$constitutive_model,
                                &perturbed_coordinates,
                            )? - nodal_forces.full_contraction(&perturbed_coordinates);
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
                        &get_helmholtz_free_energy(true, false)?,
                        &get_helmholtz_free_energy(true, true)?,
                    )
                }
                #[test]
                fn positive() -> Result<(), TestError> {
                    assert!(get_helmholtz_free_energy(true, false)? > 0.0);
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_helmholtz_free_energy(false)?,
                        &ElementNodalForcesSolid::zero(),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let minimum = get_helmholtz_free_energy(false, false)?;
                    let mut perturbed = 0.0;
                    let mut perturbed_coordinates: ElementNodalCoordinates<N> =
                        reference_coordinates().into();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_coordinates = reference_coordinates().into();
                            perturbed_coordinates[node][i] += 0.5 * EPSILON;
                            perturbed = element.helmholtz_free_energy(
                                &$constitutive_model,
                                &perturbed_coordinates,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_coordinates[node][i] -= EPSILON;
                            perturbed = element.helmholtz_free_energy(
                                &$constitutive_model,
                                &perturbed_coordinates,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(&get_helmholtz_free_energy(false, true)?, &0.0)
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq_within_tols(&get_helmholtz_free_energy(false, false)?, &0.0)
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

macro_rules! test_finite_element_with_elastic_or_hyperelastic_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        #[test]
        #[should_panic(expected = "Invalid Jacobian")]
        fn nodal_forces_invalid_jacobian() {
            let mut deformation_gradient = DeformationGradient::identity();
            deformation_gradient[0][0] = 0.0;
            get_element()
                .nodal_forces(
                    &$constitutive_model,
                    &(deformation_gradient * reference_coordinates()),
                )
                .unwrap();
        }
        #[test]
        #[should_panic(expected = "Invalid Jacobian")]
        fn nodal_stiffnesses_invalid_jacobian() {
            let mut deformation_gradient = DeformationGradient::identity();
            deformation_gradient[0][0] = 0.0;
            get_element()
                .nodal_stiffnesses(
                    &$constitutive_model,
                    &(deformation_gradient * reference_coordinates()),
                )
                .unwrap();
        }
        fn get_nodal_forces(
            is_deformed: bool,
            is_rotated: bool,
            _: bool,
        ) -> Result<ElementNodalForcesSolid<N>, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_element_transformed()
                            .nodal_forces(&$constitutive_model, &coordinates_transformed())?)
                } else {
                    Ok(get_element().nodal_forces(
                        &$constitutive_model,
                        &reference_coordinates_transformed().into(),
                    )?)
                }
            } else {
                if is_deformed {
                    Ok(get_element().nodal_forces(&$constitutive_model, &coordinates())?)
                } else {
                    Ok(get_element()
                        .nodal_forces(&$constitutive_model, &reference_coordinates().into())?)
                }
            }
        }
        fn get_nodal_stiffnesses(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<ElementNodalStiffnessesSolid<N>, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_element_transformed()
                            .nodal_stiffnesses(&$constitutive_model, &coordinates_transformed())?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, 1, 1> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_element_transformed().nodal_stiffnesses(
                            &$constitutive_model,
                            &reference_coordinates_transformed().into(),
                        )?
                        * converted)
                }
            } else {
                if is_deformed {
                    Ok(get_element().nodal_stiffnesses(&$constitutive_model, &coordinates())?)
                } else {
                    Ok(get_element()
                        .nodal_stiffnesses(&$constitutive_model, &reference_coordinates().into())?)
                }
            }
        }
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<ElementNodalStiffnessesSolid<N>, TestError> {
            let element = get_element();
            let mut finite_difference = 0.0;
            (0..N)
                .map(|a| {
                    (0..N)
                        .map(|b| {
                            (0..3)
                                .map(|i| {
                                    (0..3)
                                        .map(|j| {
                                            let mut nodal_coordinates = if is_deformed {
                                                coordinates()
                                            } else {
                                                reference_coordinates().into()
                                            };
                                            nodal_coordinates[b][j] += 0.5 * EPSILON;
                                            finite_difference = element.nodal_forces(
                                                &$constitutive_model,
                                                &nodal_coordinates,
                                            )?[a][i];
                                            nodal_coordinates[b][j] -= EPSILON;
                                            finite_difference -= element.nodal_forces(
                                                &$constitutive_model,
                                                &nodal_coordinates,
                                            )?[a][i];
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
        crate::fem::block::element::test::test_nodal_forces_and_nodal_stiffnesses!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_with_elastic_or_hyperelastic_constitutive_model;

macro_rules! test_finite_element_with_elastic_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::element::test::test_finite_element_with_elastic_or_hyperelastic_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        #[test]
        fn nodal_stiffnesses_deformed_non_symmetry() -> Result<(), TestError> {
            let nodal_stiffness = get_nodal_stiffnesses(true, false)?;
            assert!(
                assert_eq_within_tols(
                    &nodal_stiffness,
                    &(0..N)
                        .map(|a| (0..N)
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
pub(crate) use test_finite_element_with_elastic_constitutive_model;

macro_rules! test_finite_element_with_hyperelastic_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::element::test::test_finite_element_with_elastic_or_hyperelastic_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        crate::fem::block::element::test::test_helmholtz_free_energy!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_with_hyperelastic_constitutive_model;

macro_rules! test_finite_element_with_viscoelastic_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_nodal_forces(
            is_deformed: bool,
            is_rotated: bool,
            is_xtra: bool,
        ) -> Result<ElementNodalForcesSolid<N>, TestError> {
            if is_xtra {
                if is_rotated {
                    if is_deformed {
                        Ok(get_rotation_current_configuration().transpose()
                            * get_element_transformed().nodal_forces(
                                &$constitutive_model,
                                &coordinates_transformed(),
                                &velocities_transformed(),
                            )?)
                    } else {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &reference_coordinates_transformed().into(),
                            &ElementNodalVelocities::zero(),
                        )?)
                    }
                } else {
                    if is_deformed {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &coordinates(),
                            &velocities(),
                        )?)
                    } else {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &reference_coordinates().into(),
                            &ElementNodalVelocities::zero(),
                        )?)
                    }
                }
            } else {
                if is_rotated {
                    if is_deformed {
                        Ok(get_rotation_current_configuration().transpose()
                            * get_element_transformed().nodal_forces(
                                &$constitutive_model,
                                &coordinates_transformed(),
                                &ElementNodalVelocities::zero(),
                            )?)
                    } else {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &reference_coordinates_transformed().into(),
                            &ElementNodalVelocities::zero(),
                        )?)
                    }
                } else {
                    if is_deformed {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &coordinates(),
                            &ElementNodalVelocities::zero(),
                        )?)
                    } else {
                        Ok(get_element().nodal_forces(
                            &$constitutive_model,
                            &reference_coordinates().into(),
                            &ElementNodalVelocities::zero(),
                        )?)
                    }
                }
            }
        }
        fn get_nodal_stiffnesses(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<ElementNodalStiffnessesSolid<N>, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_rotation_current_configuration().transpose()
                        * get_element_transformed().nodal_stiffnesses(
                            &$constitutive_model,
                            &coordinates_transformed(),
                            &velocities_transformed(),
                        )?
                        * get_rotation_current_configuration())
                } else {
                    let converted: TensorRank2<3, 1, 1> =
                        get_rotation_reference_configuration().into();
                    Ok(converted.transpose()
                        * get_element_transformed().nodal_stiffnesses(
                            &$constitutive_model,
                            &reference_coordinates_transformed().into(),
                            &ElementNodalVelocities::zero(),
                        )?
                        * converted)
                }
            } else {
                if is_deformed {
                    Ok(get_element().nodal_stiffnesses(
                        &$constitutive_model,
                        &coordinates(),
                        &velocities(),
                    )?)
                } else {
                    Ok(get_element().nodal_stiffnesses(
                        &$constitutive_model,
                        &reference_coordinates().into(),
                        &ElementNodalVelocities::zero(),
                    )?)
                }
            }
        }
        fn get_finite_difference_of_nodal_forces(
            is_deformed: bool,
        ) -> Result<ElementNodalStiffnessesSolid<N>, TestError> {
            let element = get_element();
            let mut finite_difference = 0.0;
            (0..N)
                .map(|a| {
                    (0..N)
                        .map(|b| {
                            (0..3)
                                .map(|i| {
                                    (0..3)
                                        .map(|j| {
                                            let nodal_coordinates = if is_deformed {
                                                coordinates()
                                            } else {
                                                reference_coordinates().into()
                                            };
                                            let mut nodal_velocities = if is_deformed {
                                                velocities()
                                            } else {
                                                ElementNodalVelocities::zero()
                                            };
                                            nodal_velocities[b][j] += 0.5 * EPSILON;
                                            finite_difference = element.nodal_forces(
                                                &$constitutive_model,
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[a][i];
                                            nodal_velocities[b][j] -= EPSILON;
                                            finite_difference -= element.nodal_forces(
                                                &$constitutive_model,
                                                &nodal_coordinates,
                                                &nodal_velocities,
                                            )?[a][i];
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
        crate::fem::block::element::test::test_nodal_forces_and_nodal_stiffnesses!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_with_viscoelastic_constitutive_model;

macro_rules! test_finite_element_with_elastic_hyperviscous_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::fem::block::element::test::test_finite_element_with_viscoelastic_constitutive_model!(
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        fn get_viscous_dissipation(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<Scalar, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_element_transformed().viscous_dissipation(
                        &$constitutive_model,
                        &coordinates_transformed(),
                        &velocities_transformed(),
                    )?)
                } else {
                    Ok(get_element_transformed().viscous_dissipation(
                        &$constitutive_model,
                        &reference_coordinates_transformed().into(),
                        &ElementNodalVelocities::zero(),
                    )?)
                }
            } else {
                if is_deformed {
                    Ok(get_element().viscous_dissipation(
                        &$constitutive_model,
                        &coordinates(),
                        &velocities(),
                    )?)
                } else {
                    Ok(get_element().viscous_dissipation(
                        &$constitutive_model,
                        &reference_coordinates().into(),
                        &ElementNodalVelocities::zero(),
                    )?)
                }
            }
        }
        fn get_dissipation_potential(
            is_deformed: bool,
            is_rotated: bool,
        ) -> Result<Scalar, TestError> {
            if is_rotated {
                if is_deformed {
                    Ok(get_element_transformed().dissipation_potential(
                        &$constitutive_model,
                        &coordinates_transformed(),
                        &velocities_transformed(),
                    )?)
                } else {
                    Ok(get_element_transformed().dissipation_potential(
                        &$constitutive_model,
                        &reference_coordinates_transformed().into(),
                        &ElementNodalVelocities::zero(),
                    )?)
                }
            } else {
                if is_deformed {
                    Ok(get_element().dissipation_potential(
                        &$constitutive_model,
                        &coordinates(),
                        &velocities(),
                    )?)
                } else {
                    Ok(get_element().dissipation_potential(
                        &$constitutive_model,
                        &reference_coordinates().into(),
                        &ElementNodalVelocities::zero(),
                    )?)
                }
            }
        }
        fn get_finite_difference_of_viscous_dissipation(
            is_deformed: bool,
        ) -> Result<ElementNodalForcesSolid<N>, TestError> {
            let element = get_element();
            let mut finite_difference = 0.0;
            (0..N)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let nodal_coordinates = if is_deformed {
                                coordinates()
                            } else {
                                reference_coordinates().into()
                            };
                            let mut nodal_velocities = if is_deformed {
                                velocities()
                            } else {
                                ElementNodalVelocities::zero()
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference = element.viscous_dissipation(
                                &$constitutive_model,
                                &nodal_coordinates,
                                &nodal_velocities,
                            )?;
                            nodal_velocities[node][i] -= EPSILON;
                            finite_difference -= element.viscous_dissipation(
                                &$constitutive_model,
                                &nodal_coordinates,
                                &nodal_velocities,
                            )?;
                            Ok(finite_difference / EPSILON)
                        })
                        .collect()
                })
                .collect()
        }
        fn get_finite_difference_of_dissipation_potential(
            is_deformed: bool,
        ) -> Result<ElementNodalForcesSolid<N>, TestError> {
            let element = get_element();
            let mut finite_difference = 0.0;
            (0..N)
                .map(|node| {
                    (0..3)
                        .map(|i| {
                            let nodal_coordinates = if is_deformed {
                                coordinates()
                            } else {
                                reference_coordinates().into()
                            };
                            let mut nodal_velocities = if is_deformed {
                                velocities()
                            } else {
                                ElementNodalVelocities::zero()
                            };
                            nodal_velocities[node][i] += 0.5 * EPSILON;
                            finite_difference = element.dissipation_potential(
                                &$constitutive_model,
                                &nodal_coordinates,
                                &nodal_velocities,
                            )?;
                            nodal_velocities[node][i] -= EPSILON;
                            finite_difference -= element.dissipation_potential(
                                &$constitutive_model,
                                &nodal_coordinates,
                                &nodal_velocities,
                            )?;
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
                    assert_eq_from_fd(
                        &(get_nodal_forces(true, false, true)?
                            - get_nodal_forces(true, false, false)?),
                        &get_finite_difference_of_viscous_dissipation(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let nodal_forces = get_nodal_forces(true, false, true)?
                        - get_nodal_forces(true, false, false)?;
                    let minimum = get_viscous_dissipation(true, false)?
                        - nodal_forces.full_contraction(&velocities());
                    let mut perturbed = 0.0;
                    let mut perturbed_velocities = velocities();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = velocities();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            perturbed = element.viscous_dissipation(
                                &$constitutive_model,
                                &coordinates(),
                                &perturbed_velocities,
                            )? - nodal_forces.full_contraction(&perturbed_velocities);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_velocities[node][i] -= EPSILON;
                            perturbed = element.viscous_dissipation(
                                &$constitutive_model,
                                &coordinates(),
                                &perturbed_velocities,
                            )? - nodal_forces.full_contraction(&perturbed_velocities);
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
                        &get_viscous_dissipation(true, false)?,
                        &get_viscous_dissipation(true, true)?,
                    )
                }
                #[test]
                fn positive() -> Result<(), TestError> {
                    assert!(get_viscous_dissipation(true, false)? > 0.0);
                    Ok(())
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_viscous_dissipation(false)?,
                        &ElementNodalForcesSolid::zero(),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let minimum = get_viscous_dissipation(false, false)?;
                    let mut perturbed = 0.0;
                    let mut perturbed_velocities = ElementNodalVelocities::zero();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = ElementNodalVelocities::zero();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            perturbed = element.viscous_dissipation(
                                &$constitutive_model,
                                &reference_coordinates().into(),
                                &perturbed_velocities,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_velocities[node][i] -= EPSILON;
                            perturbed = element.viscous_dissipation(
                                &$constitutive_model,
                                &reference_coordinates().into(),
                                &perturbed_velocities,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(&get_viscous_dissipation(false, true)?, &0.0)
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(&get_viscous_dissipation(false, false)?, &0.0)
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
                        &get_nodal_forces(true, false, true)?,
                        &get_finite_difference_of_dissipation_potential(true)?,
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let nodal_forces = get_nodal_forces(true, false, true)?;
                    let minimum = get_dissipation_potential(true, false)?
                        - nodal_forces.full_contraction(&velocities());
                    let mut perturbed = 0.0;
                    let mut perturbed_velocities = velocities();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = velocities();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            perturbed = element.dissipation_potential(
                                &$constitutive_model,
                                &coordinates(),
                                &perturbed_velocities,
                            )? - nodal_forces.full_contraction(&perturbed_velocities);
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_velocities[node][i] -= EPSILON;
                            perturbed = element.dissipation_potential(
                                &$constitutive_model,
                                &coordinates(),
                                &perturbed_velocities,
                            )? - nodal_forces.full_contraction(&perturbed_velocities);
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
                        &get_dissipation_potential(true, false)?,
                        &get_dissipation_potential(true, true)?,
                    )
                }
            }
            mod undeformed {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError> {
                    assert_eq_from_fd(
                        &get_finite_difference_of_dissipation_potential(false)?,
                        &ElementNodalForcesSolid::zero(),
                    )
                }
                #[test]
                fn minimized() -> Result<(), TestError> {
                    let element = get_element();
                    let minimum = get_dissipation_potential(false, false)?;
                    let mut perturbed = 0.0;
                    let mut perturbed_velocities = ElementNodalVelocities::zero();
                    (0..N).try_for_each(|node| {
                        (0..3).try_for_each(|i| {
                            perturbed_velocities = ElementNodalVelocities::zero();
                            perturbed_velocities[node][i] += 0.5 * EPSILON;
                            perturbed = element.dissipation_potential(
                                &$constitutive_model,
                                &reference_coordinates().into(),
                                &perturbed_velocities,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            perturbed_velocities[node][i] -= EPSILON;
                            perturbed = element.dissipation_potential(
                                &$constitutive_model,
                                &reference_coordinates().into(),
                                &perturbed_velocities,
                            )?;
                            if assert_eq_within_tols(&perturbed, &minimum).is_err() {
                                assert!(perturbed > minimum)
                            }
                            Ok(())
                        })
                    })
                }
                #[test]
                fn objectivity() -> Result<(), TestError> {
                    assert_eq_within_tols(&get_dissipation_potential(false, true)?, &0.0)
                }
                #[test]
                fn zero() -> Result<(), TestError> {
                    assert_eq(&get_dissipation_potential(false, false)?, &0.0)
                }
            }
        }
    };
}
pub(crate) use test_finite_element_with_elastic_hyperviscous_constitutive_model;

macro_rules! test_finite_element_with_hyperviscoelastic_constitutive_model {
    ($element: ident, $constitutive_model: expr, $constitutive_model_type: ident) =>
    {
        crate::fem::block::element::test::test_finite_element_with_elastic_hyperviscous_constitutive_model!(
            $element, $constitutive_model, $constitutive_model_type
        );
        crate::fem::block::element::test::test_helmholtz_free_energy!(
            $element, $constitutive_model, $constitutive_model_type
        );
        #[test]
        fn dissipation_potential_deformed_positive() -> Result<(), TestError>
        {
            assert!(
                get_dissipation_potential(true, false)? > 0.0
            );
            Ok(())
        }
    }
}
pub(crate) use test_finite_element_with_hyperviscoelastic_constitutive_model;
