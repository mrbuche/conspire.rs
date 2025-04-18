macro_rules! test_hybrid_hyperelastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                Constitutive,
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::Elastic,
                    hyperelastic::{
                        ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin, NeoHookean,
                        SaintVenantKirchhoff, Yeoh, test::*,
                    },
                },
            },
            math::{Rank2, Tensor, TensorArray},
            mechanics::{
                CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffTangentStiffness,
                Scalar, SecondPiolaKirchhoffTangentStiffness,
            },
        };
        use_elastic_macros!();
        mod hybrid_0 {
            use super::*;
            test_solve!($hybrid_type::construct(
                ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_1 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_2 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                Gent::<&[Scalar; 3]>::new(GENTPARAMETERS),
                MooneyRivlin::<&[Scalar; 3]>::new(MOONEYRIVLINPARAMETERS)
            ));
        }
        mod hybrid_nested_1 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                NeoHookean::<&[Scalar; 2]>::new(NEOHOOKEANPARAMETERS),
                $hybrid_type::construct(
                    SaintVenantKirchhoff::<&[Scalar; 2]>::new(SAINTVENANTKIRCHOFFPARAMETERS),
                    Yeoh::<&[Scalar; 6]>::new(YEOHPARAMETERS)
                )
            ));
        }
        mod hybrid_nested_2 {
            use super::*;
            test_constructed_solid_hyperelastic_constitutive_model!($hybrid_type::construct(
                $hybrid_type::construct(
                    Gent::<&[Scalar; 3]>::new(GENTPARAMETERS),
                    MooneyRivlin::<&[Scalar; 3]>::new(MOONEYRIVLINPARAMETERS)
                ),
                $hybrid_type::construct(
                    NeoHookean::<&[Scalar; 2]>::new(NEOHOOKEANPARAMETERS),
                    $hybrid_type::construct(
                        SaintVenantKirchhoff::<&[Scalar; 2]>::new(SAINTVENANTKIRCHOFFPARAMETERS),
                        Yeoh::<&[Scalar; 6]>::new(YEOHPARAMETERS)
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
                Constitutive,
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::Elastic,
                    hyperelastic::{ArrudaBoyce, Fung, Gent, Hyperelastic, MooneyRivlin, test::*},
                },
            },
            math::{Rank2, Tensor, TensorArray},
            mechanics::{DeformationGradient, Scalar},
        };
        use_elastic_macros_no_tangents!();
        mod hybrid_1 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS)
            ));
        }
        mod hybrid_2 {
            use super::*;
            test_solid_hyperelastic_constitutive_model_no_tangents!($hybrid_type::construct(
                Gent::<&[Scalar; 3]>::new(GENTPARAMETERS),
                MooneyRivlin::<&[Scalar; 3]>::new(MOONEYRIVLINPARAMETERS)
            ));
        }
        crate::constitutive::hybrid::hyperelastic::test::test_panics!($hybrid_type);
        mod panic_tangents {
            use super::*;
            use crate::mechanics::test::get_deformation_gradient;
            #[test]
            #[should_panic]
            fn cauchy_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                    Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn first_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                    Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn second_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                    Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS),
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
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
            fn bulk_modulus() {
                $hybrid_type::construct(
                    ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                    Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS),
                )
                .bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn shear_modulus() {
                $hybrid_type::construct(
                    ArrudaBoyce::<&[Scalar; 3]>::new(ARRUDABOYCEPARAMETERS),
                    Fung::<&[Scalar; 4]>::new(FUNGPARAMETERS),
                )
                .shear_modulus();
            }
        }
    };
}
pub(crate) use test_panics;
