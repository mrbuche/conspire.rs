use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{PlasticFlow, RateIndependentPlastic},
        solid::{
            elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic, FirstOrderRoot},
            hyperelastic::NeoHookean,
        },
    },
    math::{
        Quantity, Rank2, Tensor, TensorArray,
        assert::{Assert, AssertionError},
        optimize::NewtonRaphson,
    },
    mechanics::DeformationGradientPlastic,
    units::{Stress, Time},
};

fn model(hardening_slope: f64) -> Canonical<NeoHookean, PlasticFlow> {
    Canonical::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        PlasticFlow {
            yield_stress: Stress::pascals(2.0),
            hardening_slope: Stress::pascals(hardening_slope),
        },
    ))
}

fn times(final_time: f64, steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Quantity::new(final_time * step as f64 / steps as f64))
        .collect()
}

fn solver() -> NewtonRaphson {
    NewtonRaphson::default()
}

fn ramp(t: Quantity<Time>) -> f64 {
    1.0 + t.value()
}

#[test]
fn stays_elastic_below_yield() -> Result<(), AssertionError> {
    let model = model(1.0);
    let (_, _, states) =
        model.root(AppliedLoad::UniaxialStress(ramp, &times(0.02, 4)), solver())?;
    states.as_slice().iter().try_for_each(|state| {
        assert_eq!(state.1.value(), 0.0);
        Assert::default().eq_within_tols(&state.0, &DeformationGradientPlastic::identity())
    })
}

#[test]
fn yields_and_returns_to_the_surface() -> Result<(), AssertionError> {
    let model = model(1.0);
    let (_, deformation_gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)),
        solver(),
    )?;
    let state = states.as_slice().last().unwrap();
    assert!(state.1.value() > 0.0);
    states
        .as_slice()
        .windows(2)
        .try_for_each(|pair| Assert::non_negative(&(pair[1].1.value() - pair[0].1.value())))?;
    Assert::default().zero_within_tols(
        &model.yield_function(
            &model
                .mandel_stress(deformation_gradients.as_slice().last().unwrap(), &state.0)?
                .deviatoric(),
            state.1,
        )?,
    )
}

#[test]
fn perfect_plasticity_caps_the_flow_stress() -> Result<(), AssertionError> {
    let model = model(0.0);
    let flow_stress = |final_time: f64| -> Result<Quantity<Stress>, AssertionError> {
        let (_, deformation_gradients, states) = model.root(
            AppliedLoad::UniaxialStress(ramp, &times(final_time, 100)),
            solver(),
        )?;
        Ok(model
            .mandel_stress(
                deformation_gradients.as_slice().last().unwrap(),
                &states.as_slice().last().unwrap().0,
            )?
            .deviatoric()
            .norm())
    };
    Assert::default().eq_within_tols(flow_stress(0.4)?, &Stress::pascals(2.0))?;
    Assert::default().eq_within_tols(flow_stress(0.5)?, &Stress::pascals(2.0))
}

#[test]
fn hardening_raises_the_flow_stress_with_plastic_strain() -> Result<(), AssertionError> {
    let model = model(1.0);
    let (_, deformation_gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)),
        solver(),
    )?;
    let state = states.as_slice().last().unwrap();
    Assert::default().eq_within_tols(
        model
            .mandel_stress(deformation_gradients.as_slice().last().unwrap(), &state.0)?
            .deviatoric()
            .norm(),
        &(Stress::pascals(2.0) + Stress::pascals(1.0) * state.1),
    )
}
