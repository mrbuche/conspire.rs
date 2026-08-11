use super::SaintVenantKirchhoff;
use crate::math::Quantity;
crate::constitutive::solid::hyperelastic_viscoplastic::test::test_model!(SaintVenantKirchhoff);

#[test]
fn root_biaxial() -> Result<(), crate::math::assert::AssertionError> {
    use crate::{
        constitutive::solid::elastic_viscoplastic::{AppliedLoad, FirstOrderRoot},
        math::{Vector, assert::Assert, integrate::DormandPrince, optimize::NewtonRaphson},
    };
    let model = SaintVenantKirchhoff {
        bulk_modulus: Quantity::new(13.0),
        shear_modulus: Quantity::new(3.0),
        yield_stress: 2.0,
        hardening_slope: 1.0,
        rate_sensitivity: 0.25,
        reference_flow_rate: 0.1,
    };
    let (times, deformation_gradients, _) = model.root(
        AppliedLoad::BiaxialStress(|t| 1.0 + t, |t| 1.0 + t / 2.0, &[0.0, 1.0]),
        DormandPrince {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..Default::default()
        },
        NewtonRaphson::default(),
    )?;
    times
        .iter()
        .zip(deformation_gradients.iter())
        .try_for_each(|(time, deformation_gradient)| {
            Assert {
                abs_tol: 1e-6,
                rel_tol: 1e-6,
                ..Default::default()
            }
            .eq_within_tols(
                Vector::from([deformation_gradient[0][0], deformation_gradient[1][1]]),
                &Vector::from([1.0 + time, 1.0 + time / 2.0]),
            )
        })
}
