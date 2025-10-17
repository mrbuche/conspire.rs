use super::super::test::*;
use super::*;

test_solid_hyperelastic_constitutive_model!(ArrudaBoyce {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    number_of_links: NUMBER_OF_LINKS,
});

mod maximum_extensibility {
    use super::*;
    #[test]
    fn cauchy_stress() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            number_of_links: 8.0,
        };
        assert_eq!(
            model.cauchy_stress(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            number_of_links: 8.0,
        };
        assert_eq!(
            model.cauchy_tangent_stiffness(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn helmholtz_free_energy_density() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            number_of_links: 8.0,
        };
        assert_eq!(
            model.helmholtz_free_energy_density(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                format!("{:?}", &model),
            ))
        )
    }
}
