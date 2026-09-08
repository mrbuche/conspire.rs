use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{PlasticFlow, RateIndependentPlastic},
        solid::{
            elastic_plastic::{
                AppliedLoad, ElasticPlastic, ElasticPlasticOrViscoplastic, FirstOrderRoot,
            },
            hyperelastic::NeoHookean,
        },
    },
    math::{
        Quantity, Rank2, Tensor, TensorArray,
        assert::{Assert, AssertionError},
        optimize::NewtonRaphson,
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic, FirstPiolaKirchhoffStress},
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

fn contract(
    tangent: &crate::mechanics::FirstPiolaKirchhoffTangentStiffness,
    direction: &DeformationGradient,
) -> FirstPiolaKirchhoffStress {
    let mut out = FirstPiolaKirchhoffStress::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += tangent[i][j][k][l].value() * direction[k][l].value();
                }
            }
            out[i][j] = Quantity::new(sum);
        }
    }
    out
}

#[test]
fn algorithmic_tangent_matches_finite_difference_through_the_return_map()
-> Result<(), AssertionError> {
    let model = model(1.0);
    // A converged plastic step: F at step 90, mapped from the state at step 89.
    let (_, deformation_gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)),
        solver(),
    )?;
    let deformation_gradient = deformation_gradients.as_slice()[90].clone();
    let previous_state = states.as_slice()[89].clone();
    assert!(
        states.as_slice()[90].1.value() > 0.0,
        "step 90 must be plastic"
    );

    let algorithmic =
        model.algorithmic_tangent_stiffness(&deformation_gradient, &previous_state)?;
    let continuum = model.first_piola_kirchhoff_tangent_stiffness(
        &deformation_gradient,
        &model.return_map(&deformation_gradient, &previous_state)?.0,
    )?;

    let step = 1.0e-5;
    let directions = [
        DeformationGradient::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.0, 0.4, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.2, 0.1, -0.15], [0.1, -0.3, 0.05], [-0.15, 0.05, 0.25]]),
    ];
    let mut continuum_departs = false;
    for direction in &directions {
        let mut plus = deformation_gradient.clone();
        let mut minus = deformation_gradient.clone();
        for k in 0..3 {
            for l in 0..3 {
                plus[k][l] += Quantity::new(step * direction[k][l].value());
                minus[k][l] -= Quantity::new(step * direction[k][l].value());
            }
        }
        let finite_difference = (model
            .first_piola_kirchhoff_stress(&plus, &model.return_map(&plus, &previous_state)?.0)?
            - model.first_piola_kirchhoff_stress(
                &minus,
                &model.return_map(&minus, &previous_state)?.0,
            )?)
            / (2.0 * step);
        Assert {
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            ..Default::default()
        }
        .eq_within_tols(contract(&algorithmic, direction), &finite_difference)?;
        if (contract(&continuum, direction) - finite_difference.clone())
            .norm()
            .value()
            > 1e-3
        {
            continuum_departs = true;
        }
    }
    // guard against a vacuous test: the plastic-corrector term must be non-negligible,
    // so the continuum (fixed-plastic-state) tangent must NOT also match the difference.
    assert!(
        continuum_departs,
        "continuum tangent matched the finite difference; test is not exercising plasticity"
    );
    Ok(())
}

#[test]
fn algorithmic_tangent_keeps_the_outer_solve_within_a_tight_step_cap() -> Result<(), AssertionError>
{
    let model = model(1.0);
    let solver = NewtonRaphson {
        max_steps: 6,
        ..Default::default()
    };
    let (_, _, states) = model.root(AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)), solver)?;
    assert!(states.as_slice().last().unwrap().1.value() > 0.0);
    Ok(())
}
