use super::super::test::*;
use super::*;
use crate::physics::molecular::single_chain::Ensemble;

mod freely_jointed_chain {
    use super::*;
    use crate::physics::molecular::single_chain::FreelyJointedChain;
    mod isometric {
        use super::*;
        test_solid_hyperelastic_constitutive_model!(EightChain {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            single_chain_model: FreelyJointedChain {
                link_length: 1.0,
                number_of_links: NUMBER_OF_LINKS as u8,
                ensemble: Ensemble::Isometric,
            }
        });
    }
    mod isotensional {
        use super::*;
        test_solid_hyperelastic_constitutive_model!(EightChain {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            single_chain_model: FreelyJointedChain {
                link_length: 1.0,
                number_of_links: NUMBER_OF_LINKS as u8,
                ensemble: Ensemble::Isotensional,
            }
        });
    }
}

// mod maximum_extensibility {
//     use super::*;
//     #[test]
//     fn cauchy_stress() {
//         let deformation_gradient =
//             DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
//         let model = EightChain {
//             bulk_modulus: BULK_MODULUS,
//             shear_modulus: SHEAR_MODULUS,
//         };
//         assert_eq!(
//             model.cauchy_stress(&deformation_gradient),
//             Err(ConstitutiveError::Custom(
//                 "Maximum extensibility reached.".to_string(),
//                 format!("{:?}", &model),
//             ))
//         )
//     }
//     #[test]
//     fn cauchy_tangent_stiffness() {
//         let deformation_gradient =
//             DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
//         let model = EightChain {
//             bulk_modulus: BULK_MODULUS,
//             shear_modulus: SHEAR_MODULUS,
//         };
//         assert_eq!(
//             model.cauchy_tangent_stiffness(&deformation_gradient),
//             Err(ConstitutiveError::Custom(
//                 "Maximum extensibility reached.".to_string(),
//                 format!("{:?}", &model),
//             ))
//         )
//     }
//     #[test]
//     fn helmholtz_free_energy_density() {
//         let deformation_gradient =
//             DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
//         let model = EightChain {
//             bulk_modulus: BULK_MODULUS,
//             shear_modulus: SHEAR_MODULUS,
//         };
//         assert_eq!(
//             model.helmholtz_free_energy_density(&deformation_gradient),
//             Err(ConstitutiveError::Custom(
//                 "Maximum extensibility reached.".to_string(),
//                 format!("{:?}", &model),
//             ))
//         )
//     }
// }
