macro_rules! test_hybrid_hyperelastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    elastic::Elastic,
                    hyperelastic::{
                        test::*, ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin, NeoHookean,
                        SaintVenantKirchoff, Yeoh,
                    },
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
        use_elastic_macros!();
        mod hybrid_0 {
            use super::*;
            test_solve!($hybrid_type::construct(
                ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                Fung::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_1 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                Fung::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_2 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                Gent::new(GENTPARAMETERS),
                MooneyRivlin::new(MOONEYRIVLINPARAMETERS)
            ));
        }
        mod hybrid_nested_1 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                NeoHookean::new(NEOHOOKEANPARAMETERS),
                $hybrid_type::construct(
                    SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS),
                    Yeoh::new(YEOHPARAMETERS)
                )
            ));
        }
        mod hybrid_nested_2 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                $hybrid_type::construct(
                    Gent::new(GENTPARAMETERS),
                    MooneyRivlin::new(MOONEYRIVLINPARAMETERS)
                ),
                $hybrid_type::construct(
                    NeoHookean::new(NEOHOOKEANPARAMETERS),
                    $hybrid_type::construct(
                        SaintVenantKirchoff::new(SAINTVENANTKIRCHOFFPARAMETERS),
                        Yeoh::new(YEOHPARAMETERS)
                    )
                )
            ));
        }
        crate::constitutive::hybrid::hyperelastic::test::test_panics!($hybrid_type);
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models;

macro_rules! test_hybrid_hyperelastic_constitutive_models_no_tangents {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    elastic::Elastic,
                    hyperelastic::{test::*, ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin},
                    Solid,
                },
                Constitutive,
            },
            math::{Rank2, Tensor, TensorArray},
            mechanics::DeformationGradient,
        };
        use_elastic_macros_no_tangents!();
        mod hybrid_1 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                Fung::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_2 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                Gent::new(GENTPARAMETERS),
                MooneyRivlin::new(MOONEYRIVLINPARAMETERS)
            ));
        }
        crate::constitutive::hybrid::hyperelastic::test::test_panics!($hybrid_type);
        mod panic_tangents {
            use super::*;
            use crate::mechanics::test::get_deformation_gradient;
            #[test]
            #[should_panic]
            fn calculate_cauchy_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                    Fung::new(FUNGPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn calculate_first_piola_kirchoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                    Fung::new(FUNGPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn calculate_second_piola_kirchoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                    Fung::new(FUNGPARAMETERS),
                )
                .calculate_cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
        }
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models_no_tangents;

macro_rules! test_panics {
    ($hybrid_type: ident) => {
        mod panic {
            use super::*;
            #[test]
            #[should_panic]
            fn get_bulk_modulus() {
                $hybrid_type::construct(
                    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                    Fung::new(FUNGPARAMETERS),
                )
                .get_bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn get_shear_modulus() {
                $hybrid_type::construct(
                    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS),
                    Fung::new(FUNGPARAMETERS),
                )
                .get_shear_modulus();
            }
            #[test]
            #[should_panic]
            fn new() {
                $hybrid_type::<ArrudaBoyce, Fung>::new(ARRUDABOYCEPARAMETERS);
            }
        }
    };
}
pub(crate) use test_panics;
