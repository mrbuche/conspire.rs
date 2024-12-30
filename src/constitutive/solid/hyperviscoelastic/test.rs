use crate::{
    constitutive::solid::elastic_hyperviscous::test::ALMANSIHAMELPARAMETERS, mechanics::Scalar,
};
pub const SAINTVENANTKIRCHOFFPARAMETERS: &[Scalar; 4] = &[
    ALMANSIHAMELPARAMETERS[0],
    ALMANSIHAMELPARAMETERS[1],
    ALMANSIHAMELPARAMETERS[2],
    ALMANSIHAMELPARAMETERS[3],
];

macro_rules! calculate_helmholtz_free_energy_density_from_deformation_gradient_simple {
    ($constitutive_model_constructed: expr, $deformation_gradient: expr) => {
        $constitutive_model_constructed
            .calculate_helmholtz_free_energy_density($deformation_gradient)
    };
}
pub(crate) use calculate_helmholtz_free_energy_density_from_deformation_gradient_simple;

macro_rules! use_elastic_hyperviscous_macros {
    () => {
        use crate::constitutive::solid::elastic_hyperviscous::test::{
            calculate_dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate,
            calculate_viscous_dissipation_from_deformation_gradient_and_deformation_gradient_rate,
            calculate_viscous_dissipation_from_deformation_gradient_rate_simple, test_solve,
            use_viscoelastic_macros,
        };
        use_viscoelastic_macros!();
    };
}
pub(crate) use use_elastic_hyperviscous_macros;

macro_rules! test_solid_hyperviscoelastic_constitutive_model
{
    ($constitutive_model: ident, $constitutive_model_parameters: expr, $constitutive_model_constructed: expr) =>
    {
        crate::constitutive::solid::elastic::test::test_solid_constitutive_construction!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        crate::constitutive::solid::hyperelastic::test::test_solid_hyperelastic_constitutive_model_no_tangents!(
            $constitutive_model_constructed
        );
        crate::constitutive::solid::viscoelastic::test::test_solid_viscous_constitutive_model!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        crate::constitutive::solid::elastic_hyperviscous::test::test_solid_elastic_hyperviscous_specifics!(
            $constitutive_model, $constitutive_model_parameters, $constitutive_model_constructed
        );
        #[test]
        fn dissipation_potential_deformed_positive() -> Result<(), TestError>
        {
            assert!(
                calculate_dissipation_potential_from_deformation_gradient_and_deformation_gradient_rate!(
                    $constitutive_model_constructed, &get_deformation_gradient(), &get_deformation_gradient_rate()
                )? > 0.0
            );
            Ok(())
        }
    }
}
pub(crate) use test_solid_hyperviscoelastic_constitutive_model;
