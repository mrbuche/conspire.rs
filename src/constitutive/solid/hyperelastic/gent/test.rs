use super::super::test::*;
use super::*;

type GentType<'a> = ArrudaBoyce<&'a [Scalar; 3]>;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model!(GentType, GENTPARAMETERS, Gent::new(GENTPARAMETERS));

test_solve!(Gent::new(GENTPARAMETERS));

#[test]
fn extensibility() {
    assert_eq!(
        &GENTPARAMETERS[2],
        Gent::new(GENTPARAMETERS).extensibility()
    )
}

mod maximum_extensibility {
    use super::*;
    #[test]
    fn cauchy_stress() {
        let deformation_gradient =
            DeformationGradient::new([[16.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]);
        let model = Gent::new(GENTPARAMETERS);
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
        let model = Gent::new(GENTPARAMETERS);
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
        let model = Gent::new(GENTPARAMETERS);
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
