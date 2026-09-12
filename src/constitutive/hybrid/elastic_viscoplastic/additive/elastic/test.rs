use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::ElasticViscoplasticAdditiveElastic,
        solid::elastic::AlmansiHamelEulerian,
        solid::elastic_viscoplastic::{AppliedLoad, RootRkmkDae},
    },
    math::{
        Quantity, Tensor, TensorArray, integrate::BogackiShampineTableau, optimize::NewtonRaphson,
    },
    mechanics::DeformationGradientPlastic,
    units::{Rate, Stress, Time},
};

// Smoke test for Phase C of the "retire the flat DAE solver into the
// field-generic driver" reframing (see memory `heterogeneous_integration`):
// a hybrid additive model gets the RKMK-DAE return map automatically, with no
// per-model StateEvolution/RootRkmkDae code, purely because it implements
// ElasticViscoplastic<Y> -- both traits are now blanket over that bound.
fn model() -> ElasticViscoplasticAdditiveElastic<
    Canonical<AlmansiHamelEulerian, ViscoplasticFlow>,
    AlmansiHamelEulerian,
    Quantity,
> {
    (
        Canonical::from((
            AlmansiHamelEulerian {
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
        AlmansiHamelEulerian {
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
fn root_rkmk_dae_keeps_the_plastic_deformation_unimodular() {
    let load = |t: Quantity<Time>| 1.0 + t.value();
    let (_, deformation_gradients, state_variables) =
        RootRkmkDae::<Quantity>::root_rkmk_dae::<BogackiShampineTableau>(
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
