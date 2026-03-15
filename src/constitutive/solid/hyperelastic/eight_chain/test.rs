use super::super::test::*;
use super::*;
use crate::physics::molecular::single_chain::Ensemble;

mod freely_jointed_chain {
    use super::*;
    use crate::physics::molecular::single_chain::FreelyJointedChain;
    // mod isometric {
    //     use super::*;
    //     const FJC: FreelyJointedChain = FreelyJointedChain {
    //         link_length: 1.0,
    //         number_of_links: NUMBER_OF_LINKS as u8,
    //         ensemble: Ensemble::Isometric,
    //     };
    //     test_solid_hyperelastic_constitutive_model!(EightChain {
    //         bulk_modulus: BULK_MODULUS,
    //         shear_modulus: SHEAR_MODULUS,
    //         single_chain_model: FJC,
    //     });
    // }
    mod isotensional {
        use super::*;
        const FJC: FreelyJointedChain = FreelyJointedChain {
            link_length: 1.0,
            number_of_links: NUMBER_OF_LINKS as u8,
            ensemble: Ensemble::Isotensional,
        };
        test_solid_hyperelastic_constitutive_model!(EightChain {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            single_chain_model: FJC,
        });
        mod consistency {
            use super::*;
            use crate::{
                constitutive::solid::hyperelastic::ArrudaBoyce, math::test::assert_eq_within_tols,
            };
            #[test]
            fn cauchy_stress() -> Result<(), TestError> {
                let eight_chain = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                let arruda_boyce = ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                };
                assert_eq_within_tols(
                    &eight_chain.cauchy_stress(&get_deformation_gradient())?,
                    &arruda_boyce.cauchy_stress(&get_deformation_gradient())?,
                )
            }
            #[test]
            fn cauchy_tangent_stiffness() -> Result<(), TestError> {
                let eight_chain = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                let arruda_boyce = ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                };
                assert_eq_within_tols(
                    &eight_chain.cauchy_tangent_stiffness(&get_deformation_gradient())?,
                    &arruda_boyce.cauchy_tangent_stiffness(&get_deformation_gradient())?,
                )
            }
            #[test]
            fn helmholtz_free_energy_density() -> Result<(), TestError> {
                let eight_chain = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                let arruda_boyce = ArrudaBoyce {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    number_of_links: NUMBER_OF_LINKS,
                };
                assert_eq_within_tols(
                    &eight_chain.helmholtz_free_energy_density(&get_deformation_gradient())?,
                    &arruda_boyce.helmholtz_free_energy_density(&get_deformation_gradient())?,
                )
            }
        }
        mod maximum_extensibility {
            use super::*;
            use crate::physics::molecular::single_chain::{Inextensible, SingleChainError};
            #[test]
            fn cauchy_stress() {
                let deformation_gradient = DeformationGradient::from([
                    [16.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.25],
                ]);
                let model = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                assert_eq!(
                    model.cauchy_stress(&deformation_gradient),
                    Err(ConstitutiveError::Upstream(
                        format!(
                            "{}",
                            SingleChainError::MaximumExtensibility(
                                format!("{:?}", FJC.maximum_nondimensional_extension()),
                                format!("{:?}", FJC),
                            )
                        ),
                        format!("{:?}", &model),
                    ))
                )
            }
            #[test]
            fn cauchy_tangent_stiffness() {
                let deformation_gradient = DeformationGradient::from([
                    [16.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.25],
                ]);
                let model = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                assert_eq!(
                    model.cauchy_tangent_stiffness(&deformation_gradient),
                    Err(ConstitutiveError::Upstream(
                        format!(
                            "{}",
                            SingleChainError::MaximumExtensibility(
                                format!("{:?}", FJC.maximum_nondimensional_extension()),
                                format!("{:?}", FJC),
                            )
                        ),
                        format!("{:?}", &model),
                    ))
                )
            }
            #[test]
            fn helmholtz_free_energy_density() {
                let deformation_gradient = DeformationGradient::from([
                    [16.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.25],
                ]);
                let model = EightChain {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    single_chain_model: FJC,
                };
                assert_eq!(
                    model.helmholtz_free_energy_density(&deformation_gradient),
                    Err(ConstitutiveError::Upstream(
                        format!(
                            "{}",
                            SingleChainError::MaximumExtensibility(
                                format!("{:?}", FJC.maximum_nondimensional_extension()),
                                format!("{:?}", FJC),
                            )
                        ),
                        format!("{:?}", &model),
                    ))
                )
            }
        }
    }
}
