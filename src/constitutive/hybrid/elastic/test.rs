macro_rules! test_hybrid_elastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                Constitutive,
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::{AlmansiHamel, Elastic, test::*},
                    hyperelastic::{NeoHookean, test::NEOHOOKEANPARAMETERS},
                },
            },
            math::{Rank2, TensorArray},
            mechanics::{
                CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffTangentStiffness,
                SecondPiolaKirchhoffTangentStiffness,
            },
        };
        mod hybrid_1 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                NeoHookean::new(NEOHOOKEANPARAMETERS)
            ));
        }
        mod hybrid_nested_1 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS)
                )
            ));
        }
        mod hybrid_nested_2 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                $hybrid_type::construct(
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS)
                ),
                $hybrid_type::construct(
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                    $hybrid_type::construct(
                        AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                        NeoHookean::new(NEOHOOKEANPARAMETERS)
                    )
                )
            ));
        }
        crate::constitutive::hybrid::elastic::test::test_panics!($hybrid_type);
    };
}
pub(crate) use test_hybrid_elastic_constitutive_models;

macro_rules! test_hybrid_elastic_constitutive_models_no_tangents {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                Constitutive,
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::{AlmansiHamel, Elastic, test::*},
                    hyperelastic::{NeoHookean, test::NEOHOOKEANPARAMETERS},
                },
            },
            math::{Rank2, TensorArray},
            mechanics::DeformationGradient,
        };
        mod hybrid_1 {
            use super::*;
            test_solid_constitutive_model_no_tangents!($hybrid_type::construct(
                AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                NeoHookean::new(NEOHOOKEANPARAMETERS)
            ));
        }
        crate::constitutive::hybrid::elastic::test::test_panics!($hybrid_type);
        mod panic_tangents {
            use super::*;
            use crate::mechanics::test::get_deformation_gradient;
            #[test]
            #[should_panic]
            fn cauchy_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn first_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn second_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
        }
    };
}
pub(crate) use test_hybrid_elastic_constitutive_models_no_tangents;

macro_rules! test_panics {
    ($hybrid_type: ident) => {
        mod panic {
            use super::*;
            #[test]
            #[should_panic]
            fn bulk_modulus() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn shear_modulus() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .shear_modulus();
            }
            #[test]
            #[should_panic]
            fn new() {
                $hybrid_type::<AlmansiHamel, NeoHookean>::new(ALMANSIHAMELPARAMETERS);
            }
        }
    };
}
pub(crate) use test_panics;
