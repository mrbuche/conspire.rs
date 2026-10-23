use super::Root;
use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{Linear, PlasticFlow, VonMises},
        solid::{elastic_plastic::AppliedLoad, hyperelastic::NeoHookean},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
        optimize::{NewtonRaphson, SolveStrategy},
    },
    units::{Stress, Time},
};

const TOLERANCE: Quantity = Quantity::new(1e-3);

type Mises = PlasticFlow<VonMises, Linear>;

fn model(hardening_slope: f64) -> Canonical<NeoHookean, Mises> {
    Canonical::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        PlasticFlow {
            surface: VonMises,
            hardening: Linear {
                yield_stress: Stress::pascals(2.0),
                hardening_slope: Stress::pascals(hardening_slope),
            },
        },
    ))
}

fn solver() -> NewtonRaphson {
    NewtonRaphson::default()
}

fn ramp(t: Quantity<Time>) -> f64 {
    1.0 + t.value()
}

fn times(final_time: f64, steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Quantity::new(final_time * step as f64 / steps as f64))
        .collect()
}

fn root(
    model: &Canonical<NeoHookean, Mises>,
    applied_load: AppliedLoad,
    solver: NewtonRaphson,
) -> Result<
    (
        crate::mechanics::Times,
        crate::mechanics::DeformationGradients,
        crate::constitutive::fluid::plastic::PlasticStateVariablesHistory,
    ),
    crate::constitutive::ConstitutiveError,
> {
    model.root(
        applied_load,
        solver,
        SolveStrategy::Condensed(NewtonRaphson::default()),
        TOLERANCE,
    )
}

fn span(final_time: f64) -> [Quantity<Time>; 2] {
    [Quantity::new(0.0), Quantity::new(final_time)]
}

fn turning(t: Quantity<Time>) -> f64 {
    1.0 + 0.6 * t.value() * t.value()
}

fn final_strain(
    model: &Canonical<NeoHookean, Mises>,
    load: AppliedLoad,
) -> Result<f64, crate::constitutive::ConstitutiveError> {
    let (_, _, states) = root(model, load, solver())?;
    Ok(states.as_slice().last().unwrap().1.value())
}

#[test]
fn proportional_loading_resolves_the_onset_then_takes_large_steps() -> Result<(), AssertionError> {
    let model = model(1.0);
    let (accepted, gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &span(1.0)),
        solver(),
        SolveStrategy::Condensed(NewtonRaphson::default()),
        TOLERANCE,
    )?;
    let strains = states
        .as_slice()
        .iter()
        .map(|state| state.1.value())
        .collect::<Vec<_>>();
    let first_plastic = strains
        .iter()
        .copied()
        .find(|&strain| strain > 0.0)
        .unwrap();
    let yield_strain = 2.0 / (3.0 * 3.0);
    assert!(
        first_plastic <= TOLERANCE.value() * yield_strain,
        "{first_plastic}"
    );
    assert!(accepted.len() <= 20, "{} steps", accepted.len());
    let (_, fine_gradients, fine_states) = root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(1.0, 200)),
        solver(),
    )?;
    let assert = Assert {
        abs_tol: 1e-8,
        rel_tol: 1e-8,
        ..Default::default()
    };
    assert.eq_within_tols(
        gradients.as_slice().last().unwrap(),
        fine_gradients.as_slice().last().unwrap(),
    )?;
    assert.eq_within_tols(
        states.as_slice().last().unwrap().1,
        &fine_states.as_slice().last().unwrap().1,
    )?;
    Ok(())
}

#[test]
fn a_turning_load_takes_more_steps_as_the_tolerance_tightens() -> Result<(), AssertionError> {
    let model = model(1.0);
    let reference = final_strain(
        &model,
        AppliedLoad::BiaxialStress(ramp, turning, &times(1.0, 800)),
    )?;
    let mut counts = Vec::new();
    let mut errors = Vec::new();
    for tolerance in [1e-2, 1e-3, 1e-4].map(Quantity::new) {
        let (accepted, _, states) = model.root(
            AppliedLoad::BiaxialStress(ramp, turning, &span(1.0)),
            solver(),
            SolveStrategy::Condensed(NewtonRaphson::default()),
            tolerance,
        )?;
        counts.push(accepted.len());
        errors.push((states.as_slice().last().unwrap().1.value() - reference).abs());
    }
    assert!(counts[0] < counts[1] && counts[1] < counts[2], "{counts:?}");
    assert!(errors[2] < errors[0], "{errors:?}");
    assert!(counts[1] > 12, "{counts:?}");
    Ok(())
}

#[test]
fn steps_end_on_every_requested_time_and_only_there() -> Result<(), AssertionError> {
    let model = model(1.0);
    let requested = [0.0, 0.3, 0.55, 1.0].map(Quantity::<Time>::new);
    let (accepted, gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &requested),
        solver(),
        SolveStrategy::Condensed(NewtonRaphson::default()),
        TOLERANCE,
    )?;
    assert_eq!(
        accepted.iter().map(|t| t.value()).collect::<Vec<_>>(),
        vec![0.0, 0.3, 0.55, 1.0]
    );
    assert_eq!(gradients.len(), 4);
    assert_eq!(states.len(), 4);
    Ok(())
}

#[test]
fn strategies_agree() -> Result<(), AssertionError> {
    let model = model(1.0);
    let run = |strategy| {
        model.root(
            AppliedLoad::BiaxialStress(ramp, turning, &span(1.0)),
            solver(),
            strategy,
            TOLERANCE,
        )
    };
    let (_, gradients, states) = run(SolveStrategy::Condensed(NewtonRaphson::default()))?;
    let (_, gradients_monolithic, states_monolithic) =
        run(SolveStrategy::Monolithic { elimination: true })?;
    let assert = Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Default::default()
    };
    assert.eq_within_tols(
        gradients.as_slice().last().unwrap(),
        gradients_monolithic.as_slice().last().unwrap(),
    )?;
    assert.eq_within_tols(
        states.as_slice().last().unwrap().1,
        &states_monolithic.as_slice().last().unwrap().1,
    )?;
    Ok(())
}

#[test]
fn the_times_must_be_at_least_two_and_increasing() {
    let model = model(1.0);
    for times in [vec![0.0], vec![1.0, 0.0], vec![0.5, 0.5]] {
        let times = times
            .into_iter()
            .map(Quantity::<Time>::new)
            .collect::<Vec<_>>();
        assert!(
            model
                .root(
                    AppliedLoad::UniaxialStress(ramp, &times),
                    solver(),
                    SolveStrategy::Condensed(NewtonRaphson::default()),
                    TOLERANCE,
                )
                .is_err()
        );
    }
}

#[test]
fn the_error_of_the_result_falls_more_slowly_than_the_tolerance() -> Result<(), AssertionError> {
    use crate::math::Tensor;
    let model = model(1.0);
    let turning = |t: Quantity<Time>| 1.0 + 0.6 * t.value() * t.value();
    let run = |load: AppliedLoad, tolerance: f64| {
        model.root(
            load,
            solver(),
            SolveStrategy::Condensed(NewtonRaphson::default()),
            Quantity::new(tolerance),
        )
    };
    let (_, _, reference) = run(
        AppliedLoad::BiaxialStress(ramp, turning, &times(1.0, 500)),
        1e9,
    )?;
    let reference = reference.as_slice().last().unwrap().clone();
    let span = [Quantity::new(0.0), Quantity::new(1.0)];
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    for tolerance in [1e-2, 1e-4] {
        let (accepted, _, states) =
            run(AppliedLoad::BiaxialStress(ramp, turning, &span), tolerance)?;
        let last = states.as_slice().last().unwrap();
        errors.push((last.0.clone() - reference.0.clone()).norm().value());
        steps.push(accepted.len() - 1);
    }
    assert!(errors[1] < errors[0], "{errors:?}");
    assert!(steps[1] > steps[0], "{steps:?}");
    let ratio = errors[0] / errors[1];
    assert!((4.0..30.0).contains(&ratio), "{ratio} {errors:?}");
    Ok(())
}

#[test]
fn the_yield_strain_is_the_yield_stress_over_three_shear_moduli() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::ElasticPlastic,
        math::{TensorArray, TensorTuple},
        mechanics::DeformationGradientPlastic,
    };
    let model = model(1.0);
    let state = |strain: f64| {
        TensorTuple(
            DeformationGradientPlastic::identity(),
            Quantity::new(strain),
        )
    };
    let assert = Assert::default();
    assert.eq_within_tols(model.yield_strain(&state(0.0))?.value(), 2.0 / 9.0)?;
    assert.eq_within_tols(model.yield_strain(&state(0.5))?.value(), 2.5 / 9.0)
}
