macro_rules! test_hybrid_elastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    elastic::{test::*, AlmansiHamel, Elastic},
                    hyperelastic::{test::NEOHOOKEANPARAMETERS, NeoHookean},
                    Solid,
                },
                Constitutive,
            },
            math::{Rank2, Tensor, TensorArray},
            mechanics::{
                CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchoffTangentStiffness,
                SecondPiolaKirchoffTangentStiffness,
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
                hybrid::Hybrid,
                solid::{
                    elastic::{test::*, AlmansiHamel, Elastic},
                    hyperelastic::{test::NEOHOOKEANPARAMETERS, NeoHookean},
                    Solid,
                },
                Constitutive,
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
            fn calculate_cauchy_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn calculate_first_piola_kirchoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn calculate_second_piola_kirchoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
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
            fn get_bulk_modulus() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .get_bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn get_shear_modulus() {
                $hybrid_type::construct(
                    AlmansiHamel::new(ALMANSIHAMELPARAMETERS),
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                )
                .get_shear_modulus();
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
