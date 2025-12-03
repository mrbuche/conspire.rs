macro_rules! test_hybrid_hyperelastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::constitutive::solid::hyperelastic::{ArrudaBoyce, Fung, test::*};
        test_solid_hyperelastic_constitutive_model!($hybrid_type::from((
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
        )));
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models;

macro_rules! test_hybrid_hyperelastic_constitutive_models_no_tangents {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::solid::{
                Solid,
                elastic::Elastic,
                hyperelastic::{ArrudaBoyce, Fung, Hyperelastic, test::*},
            },
            math::{Rank2, Tensor},
            mechanics::DeformationGradient,
        };
        use_elastic_macros_no_tangents!();
        test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::from((
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
        )));
        mod panic_tangents {
            use super::*;
            use crate::mechanics::test::get_deformation_gradient;
            #[test]
            #[should_panic]
            fn cauchy_tangent_stiffness() {
                $hybrid_type::from((
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
                ))
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn first_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::from((
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
                ))
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn second_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::from((
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
                ))
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn bulk_modulus() {
                $hybrid_type::from((
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
                ))
                .bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn shear_modulus() {
                $hybrid_type::from((
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
                ))
                .shear_modulus();
            }
        }
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models_no_tangents;
