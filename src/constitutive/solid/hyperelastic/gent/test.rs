use super::super::test::*;
use super::*;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(Gent, GENTPARAMETERS, Gent::new(GENTPARAMETERS));

test_solve!(Gent::new(GENTPARAMETERS));

#[test]
fn get_extensibility() {
    assert_eq!(
        &GENTPARAMETERS[2],
        Gent::new(GENTPARAMETERS).get_extensibility()
    )
}

mod maximum_extensibility {
    use super::*;
    #[test]
    fn calculate_cauchy_stress() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = Gent::new(GENTPARAMETERS);
        assert_eq!(
            model.calculate_cauchy_stress(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.copy(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn calculate_cauchy_tangent_stiffness() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = Gent::new(GENTPARAMETERS);
        assert_eq!(
            model.calculate_cauchy_tangent_stiffness(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.copy(),
                format!("{:?}", &model),
            ))
        )
    }
    #[test]
    fn calculate_helmholtz_free_energy_density() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = Gent::new(GENTPARAMETERS);
        assert_eq!(
            model.calculate_helmholtz_free_energy_density(&deformation_gradient),
            Err(ConstitutiveError::Custom(
                "Maximum extensibility reached.".to_string(),
                deformation_gradient.copy(),
                format!("{:?}", &model),
            ))
        )
    }
}
