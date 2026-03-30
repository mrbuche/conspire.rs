use super::super::test::*;
use super::*;

const ARRUDABOYCE: ArrudaBoyce = ArrudaBoyce {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    number_of_links: 8.0,
};

test_solid_hyperelastic_constitutive_model!(ARRUDABOYCE);

mod maximum_extensibility {
    use super::*;
    #[test]
    fn cauchy_stress() {
        let deformation_gradient =
            DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ARRUDABOYCE;
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
            DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ARRUDABOYCE;
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
            DeformationGradient::from([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ARRUDABOYCE;
        assert_eq!(
            model.helmholtz_free_energy_density(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                format!("{:?}", &model),
            ))
        )
    }
}
