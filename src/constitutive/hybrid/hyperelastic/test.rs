macro_rules! test_hybrid_hyperelastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::Elastic,
                    hyperelastic::{
                        ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin, NeoHookean,
                        SaintVenantKirchhoff, test::*,
                    },
                },
            },
            math::{Rank2, Tensor},
            mechanics::{CauchyTangentStiffness, DeformationGradient},
        };
        use_elastic_macros!();
        mod hybrid_0 {
            use super::*;
            test_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                },
                Fung {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    exponent: EXPONENT,
                    extra_modulus: EXTRA_MODULUS,
                }
            ));
        }
        mod hybrid_1 {
            use super::*;
            test_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                },
                Fung {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    exponent: EXPONENT,
                    extra_modulus: EXTRA_MODULUS,
                }
            ));
        }
        // mod hybrid_2 {
        //     use super::*;
        //     test_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
        //         Gent {
        //             bulk_modulus: BULK_MODULUS,
        //             shear_modulus: SHEAR_MODULUS,
        //             extensibility: EXTENSIBILITY,
        //         },
        //         MooneyRivlin {
        //             bulk_modulus: BULK_MODULUS,
        //             shear_modulus: SHEAR_MODULUS,
        //             extra_modulus: EXTRA_MODULUS,
        //         }
        //     ));
        // }
        mod hybrid_nested_1 {
            use super::*;
            test_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                NeoHookean {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                },
                $hybrid_type::construct(
                    SaintVenantKirchhoff {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    Fung {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        exponent: EXPONENT,
                        extra_modulus: EXTRA_MODULUS,
                    }
                )
            ));
        }
        mod hybrid_nested_2 {
            use super::*;
            test_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                $hybrid_type::construct(
                    Gent {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        extensibility: EXTENSIBILITY,
                    },
                    MooneyRivlin {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        extra_modulus: EXTRA_MODULUS,
                    }
                ),
                $hybrid_type::construct(
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    $hybrid_type::construct(
                        SaintVenantKirchhoff {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        Fung {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            exponent: EXPONENT,
                            extra_modulus: EXTRA_MODULUS,
                        }
                    )
                )
            ));
        }
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models;

macro_rules! test_hybrid_hyperelastic_constitutive_models_no_tangents {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::Elastic,
                    hyperelastic::{ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin, test::*},
                },
            },
            math::{Rank2, Tensor},
            mechanics::DeformationGradient,
        };
        use_elastic_macros_no_tangents!();
        mod hybrid_1 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                },
                Fung {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    exponent: EXPONENT,
                    extra_modulus: EXTRA_MODULUS,
                }
            ));
        }
        mod hybrid_2 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                Gent {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    extensibility: EXTENSIBILITY,
                },
                MooneyRivlin {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    extra_modulus: EXTRA_MODULUS,
                }
            ));
        }
        mod panic_tangents {
            use super::*;
            use crate::mechanics::test::get_deformation_gradient;
            #[test]
            #[should_panic]
            fn cauchy_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        number_of_links: NUMBER_OF_LINKS,
                    },
                    Fung {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        exponent: EXPONENT,
                        extra_modulus: EXTRA_MODULUS,
                    },
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn first_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        number_of_links: NUMBER_OF_LINKS,
                    },
                    Fung {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        exponent: EXPONENT,
                        extra_modulus: EXTRA_MODULUS,
                    },
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn second_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        number_of_links: NUMBER_OF_LINKS,
                    },
                    Fung {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        exponent: EXPONENT,
                        extra_modulus: EXTRA_MODULUS,
                    },
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
        }
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models_no_tangents;
