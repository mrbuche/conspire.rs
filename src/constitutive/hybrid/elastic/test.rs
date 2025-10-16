macro_rules! test_hybrid_elastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::{
            constitutive::{
                hybrid::Hybrid,
                solid::{
                    Solid,
                    elastic::{AlmansiHamel, Elastic, test::*},
                    hyperelastic::NeoHookean,
                },
            },
            math::Rank2,
            mechanics::{CauchyTangentStiffness, DeformationGradient},
        };
        mod hybrid_1 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                AlmansiHamel {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                },
                NeoHookean {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                },
            ));
        }
        mod hybrid_nested_1 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                AlmansiHamel {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                },
                $hybrid_type::construct(
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    }
                )
            ));
        }
        mod hybrid_nested_2 {
            use super::*;
            test_constructed_solid_constitutive_model!($hybrid_type::construct(
                $hybrid_type::construct(
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    }
                ),
                $hybrid_type::construct(
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    $hybrid_type::construct(
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        },
                        NeoHookean {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                        }
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
                    Solid,
                    elastic::{AlmansiHamel, Elastic, test::*},
                    hyperelastic::NeoHookean,
                },
            },
            math::Rank2,
            mechanics::DeformationGradient,
        };
        mod hybrid_1 {
            use super::*;
            test_solid_constitutive_model_no_tangents!($hybrid_type::construct(
                AlmansiHamel {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                },
                NeoHookean {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                }
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
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn first_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                )
                .cauchy_tangent_stiffness(&get_deformation_gradient())
                .unwrap();
            }
            #[test]
            #[should_panic]
            fn second_piola_kirchhoff_tangent_stiffness() {
                $hybrid_type::construct(
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
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
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                )
                .bulk_modulus();
            }
            #[test]
            #[should_panic]
            fn shear_modulus() {
                $hybrid_type::construct(
                    AlmansiHamel {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                )
                .shear_modulus();
            }
        }
    };
}
pub(crate) use test_panics;
