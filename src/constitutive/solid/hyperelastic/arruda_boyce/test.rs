use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(
    ArrudaBoyce,
    ARRUDABOYCEPARAMETERS,
    ArrudaBoyce::new(ARRUDABOYCEPARAMETERS)
);

test_solve!(ArrudaBoyce::new(ARRUDABOYCEPARAMETERS));

#[test]
fn number_of_links() {
    assert_eq!(
        &ARRUDABOYCEPARAMETERS[2],
        ArrudaBoyce::new(ARRUDABOYCEPARAMETERS).number_of_links()
    )
}

mod maximum_extensibility {
    use super::*;
    #[test]
    fn cauchy_stress() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce::new(ARRUDABOYCEPARAMETERS);
        assert_eq!(
            model.cauchy_stress(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn cauchy_tangent_stiffness() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce::new(ARRUDABOYCEPARAMETERS);
        assert_eq!(
            model.cauchy_tangent_stiffness(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn helmholtz_free_energy_density() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = ArrudaBoyce::new(ARRUDABOYCEPARAMETERS);
        assert_eq!(
            model.helmholtz_free_energy_density(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.clone(),
                format!("{:?}", &model),
            ))
        )
    }
}
