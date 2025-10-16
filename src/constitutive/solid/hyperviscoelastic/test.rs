pub use crate::constitutive::solid::elastic_hyperviscous::test::{
    BULK_MODULUS, BULK_VISCOSITY, SHEAR_MODULUS, SHEAR_VISCOSITY,
};

macro_rules! helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model: expr, $deformation_gradient: expr) => {
        $constitutive_model.helmholtz_free_energy_density($deformation_gradient)
    };
}
pub(crate) use helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_elastic_hyperviscous_macros {
    () => {
        use crate::constitutive::solid::elastic_hyperviscous::test::{
            dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate,
            use_viscoelastic_macros,
            viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate,
            viscous_dissipation_from_deformation_gradient_rate_simple,
        };
        use_viscoelastic_macros!();
    };
}
pub(crate) use use_elastic_hyperviscous_macros;

macro_rules! test_solid_hyperviscoelastic_constitutive_model
{
    ($constitutive_model: expr) =>
    {
        use_elastic_hyperviscous_macros!();
        crate::constitutive::solid::elastic::test::test_solid_constitutive!(
            $constitutive_model
        );
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_tangents!(
            $constitutive_model
        );
        crate::constitutive::solid::viscoelastic::test::test_solid_viscous_constitutive_model!(
            $constitutive_model
        );
        crate::constitutive::solid::elastic_hyperviscous::test::test_solid_elastic_hyperviscous_specifics!(
            $constitutive_model
        );
        #[test]
        fn dissipation_potential_deformed_positive() -> Result<(), TestError>
        {
            assert!(
                dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                    $constitutive_model, &get_deformation_gradient(), &get_deformation_gradient_rate()
                )? > 0.0
            );
            Ok(())
        }
    }
}
pub(crate) use test_solid_hyperviscoelastic_constitutive_model;
