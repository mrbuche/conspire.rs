use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::ElasticViscoplasticAdditiveElastic,
        solid::{
            elastic_viscoplastic::AppliedLoad, hyperelastic::Hencky,
            hyperelastic_viscoplastic::RootRkmkDaeMinimize,
        },
    },
    math::{
        Quantity, Tensor, TensorArray, integrate::BogackiShampineTableau, optimize::NewtonRaphson,
    },
    mechanics::DeformationGradientPlastic,
    units::{Rate, Stress, Time},
};

// Smoke test for Phase C of the "retire the flat DAE solver into the
// field-generic driver" reframing (see memory `heterogeneous_integration`):
// the same hybrid additive struct also gets the minimize-based RKMK-DAE
// return map automatically, purely because it implements
// HyperelasticViscoplastic<Y>.
fn model()
-> ElasticViscoplasticAdditiveElastic<Canonical<Hencky, ViscoplasticFlow>, Hencky, Quantity> {
    (
        Canonical::from((
            Hencky {
                bulk_modulus: Stress::pascals(13.0),
                shear_modulus: Stress::pascals(3.0),
            },
            ViscoplasticFlow {
                yield_stress: Stress::pascals(2.0),
                hardening_slope: Stress::pascals(1.0),
                rate_sensitivity: 0.25,
                reference_flow_rate: Rate::per_second(0.1),
            },
        )),
        Hencky {
            bulk_modulus: Stress::pascals(5.0),
            shear_modulus: Stress::pascals(1.0),
        },
    )
        .into()
}

fn time(steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|i| Quantity::new(i as f64 / steps as f64))
        .collect()
}

#[test]
fn root_rkmk_dae_minimize_keeps_the_plastic_deformation_unimodular() {
    let load = |t: Quantity<Time>| 1.0 + t.value();
    let (_, deformation_gradients, state_variables) =
        RootRkmkDaeMinimize::<Quantity>::root_rkmk_dae_minimize::<BogackiShampineTableau>(
            &model(),
            AppliedLoad::UniaxialStress(load, &time(20)),
            NewtonRaphson::default(),
        )
        .unwrap();
    assert_eq!(deformation_gradients.iter().count(), 21);
    let deformation_gradient_p = &state_variables.iter().last().unwrap().0;
    assert!((deformation_gradient_p.determinant() - 1.0).abs() < 1e-10);
    // and F_p actually flowed
    assert!(
        (deformation_gradient_p - &DeformationGradientPlastic::identity())
            .norm()
            .value()
            > 1e-3
    );
}
