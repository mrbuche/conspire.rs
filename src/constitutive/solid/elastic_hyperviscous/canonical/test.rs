use super::*;
use crate::{
    constitutive::{
        fluid::{hyperviscous::Newtonian, viscous::Viscous},
        solid::{
            elastic::AlmansiHamelEulerian, elastic_hyperviscous::test::*,
            viscoelastic::Viscoelastic,
        },
    },
    math::{Rank2, Tensor, assert::Assert},
    mechanics::{
        CauchyRateTangentStiffness, DeformationGradient, DeformationGradientRate,
        FirstPiolaKirchhoffRateTangentStiffness, SecondPiolaKirchhoffRateTangentStiffness,
    },
};

fn model() -> Canonical<AlmansiHamelEulerian, Newtonian> {
    Canonical::from((
        AlmansiHamelEulerian {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        Newtonian {
            bulk_viscosity: BULK_VISCOSITY,
            shear_viscosity: SHEAR_VISCOSITY,
        },
    ))
}

test_solid_elastic_hyperviscous_constitutive_model!(model());

mod consistency {
    use super::*;
    use crate::{constitutive::solid::elastic::Elastic, mechanics::test::get_deformation_gradient};
    #[test]
    fn cauchy_stress() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &model().cauchy_stress(
                &get_deformation_gradient(),
                &DeformationGradientRate::zero(),
            )?,
            &AlmansiHamelEulerian {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            }
            .cauchy_stress(&get_deformation_gradient())?,
        )
    }
}
