use super::super::test::*;
use super::*;

use crate::{
    math::Tensor,
    mechanics::{
        CauchyRateTangentStiffness, CauchyTangentStiffness,
        FirstPiolaKirchhoffRateTangentStiffness, FirstPiolaKirchhoffTangentStiffness,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

test_solid_hyperviscoelastic_constitutive_model!(SaintVenantKirchhoff {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    bulk_viscosity: BULK_VISCOSITY,
    shear_viscosity: SHEAR_VISCOSITY,
});

mod consistency {
    use super::*;
    use crate::{
        constitutive::solid::{
            elastic::Elastic,
            hyperelastic::{
                Hyperelastic, SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
            },
        },
        math::test::assert_eq_within_tols,
    };
    #[test]
    fn helmholtz_free_energy_density() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            bulk_viscosity: BULK_VISCOSITY,
            shear_viscosity: SHEAR_VISCOSITY,
        };
        let hyperelastic_model = HyperelasticSaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.helmholtz_free_energy_density(&get_deformation_gradient())?,
            &hyperelastic_model.helmholtz_free_energy_density(&get_deformation_gradient())?,
        )
    }
    #[test]
    fn cauchy_stress() -> Result<(), TestError> {
        let model = SaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            bulk_viscosity: BULK_VISCOSITY,
            shear_viscosity: SHEAR_VISCOSITY,
        };
        let hyperelastic_model = HyperelasticSaintVenantKirchhoff {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        assert_eq_within_tols(
            &model.cauchy_stress(
                &get_deformation_gradient(),
                &DeformationGradientRate::zero(),
            )?,
            &hyperelastic_model.cauchy_stress(&get_deformation_gradient())?,
        )
    }
}
