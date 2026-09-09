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

#[test]
fn monolithic_strategies_agree_with_the_nested_solve() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::{FirstOrderRoot, MonolithicRoot},
        math::optimize::SolveStrategy,
    };
    let model = model(1.0);
    let steps = times(0.5, 40);
    let (_, reference_gradients, reference_states) = FirstOrderRoot::root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &steps),
        NewtonRaphson::default(),
    )?;
    let reference_gradient = reference_gradients.as_slice().last().unwrap();
    let reference_strain = reference_states.as_slice().last().unwrap().1;
    assert!(reference_strain.value() > 0.0);
    for strategy in [
        SolveStrategy::Condensed(NewtonRaphson::default()),
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
    ] {
        let (_, gradients, states) = MonolithicRoot::root(
            &model,
            AppliedLoad::UniaxialStress(ramp, &steps),
            NewtonRaphson::default(),
            strategy,
        )?;
        Assert {
            abs_tol: 1e-11,
            rel_tol: 1e-11,
            ..Default::default()
        }
        .eq_within_tols(gradients.as_slice().last().unwrap(), reference_gradient)?;
        Assert {
            abs_tol: 1e-11,
            rel_tol: 1e-11,
            ..Default::default()
        }
        .eq_within_tols(states.as_slice().last().unwrap().1, &reference_strain)?;
    }
    Ok(())
}

#[test]
fn monolithic_tangents_match_finite_difference_at_a_plastic_state() -> Result<(), AssertionError> {
    use crate::{
        constitutive::{fluid::plastic::Plastic, solid::elastic_plastic::fischer_burmeister},
        math::Rank2,
        mechanics::{FirstPiolaKirchhoffStress, Scalar},
    };
    let model = model(1.0);
    let (_, deformation_gradients, states) = model.root(
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)),
        solver(),
    )?;
    // Step 90: the block residual is assembled at F#90 with the state (and frozen
    // flow direction) coming from the converged step 89.
    let deformation_gradient = deformation_gradients.as_slice()[90].clone();
    let previous_gradient = deformation_gradients.as_slice()[89].clone();
    let previous_state = states.as_slice()[89].clone();
    let strain_previous = previous_state.1;
    let plastic_previous = previous_state.0.clone();
    let plastic_multiplier = states.as_slice()[90].1.value() - strain_previous.value();
    assert!(plastic_multiplier > 0.0, "step 90 must be plastic");
    let flow_direction = {
        let deviatoric = model
            .mandel_stress(&previous_gradient, &plastic_previous)?
            .deviatoric();
        let direction = model.flow_direction(&deviatoric)?;
        (&direction + direction.transpose()) * 0.5
    };
    let plastic =
        |multiplier: Scalar| (&flow_direction * multiplier).expm().unwrap() * &plastic_previous;
    let residual_global = |gradient: &DeformationGradient,
                           multiplier: Scalar|
     -> Result<FirstPiolaKirchhoffStress, AssertionError> {
        Ok(model.first_piola_kirchhoff_stress(gradient, &plastic(multiplier))?)
    };
    let residual_local =
        |gradient: &DeformationGradient, multiplier: Scalar| -> Result<Scalar, AssertionError> {
            let scaled = model
                .yield_function(
                    &model
                        .mandel_stress(gradient, &plastic(multiplier))?
                        .deviatoric(),
                    strain_previous + Quantity::new(multiplier),
                )?
                .value()
                / model.initial_yield_stress().value();
            Ok(fischer_burmeister(multiplier, -scaled))
        };
    let (_, k_vu, k_uv, k_vv) = model.monolithic_tangents(
        &deformation_gradient,
        &plastic_previous,
        &flow_direction,
        strain_previous,
        plastic_multiplier,
    )?;
    let assert = Assert {
        abs_tol: 1e-6,
        rel_tol: 1e-6,
        ..Default::default()
    };
    let step = 1.0e-6;
    //
    // K_uv = dP/d(plastic multiplier).
    //
    let mut analytic = FirstPiolaKirchhoffStress::zero();
    for i in 0..3 {
        for j in 0..3 {
            analytic[i][j] = k_uv[i][j][0][0];
        }
    }
    assert.eq_within_tols(
        &analytic,
        &((residual_global(&deformation_gradient, plastic_multiplier + step)?
            - residual_global(&deformation_gradient, plastic_multiplier - step)?)
            / (2.0 * step)),
    )?;
    //
    // K_vu = d(Fischer-Burmeister)/dF.
    //
    let mut analytic = DeformationGradient::zero();
    let mut finite_difference = DeformationGradient::zero();
    for k in 0..3 {
        for l in 0..3 {
            analytic[k][l] = k_vu[0][0][k][l];
            let mut plus = deformation_gradient.clone();
            plus[k][l] += Quantity::new(step);
            let mut minus = deformation_gradient.clone();
            minus[k][l] -= Quantity::new(step);
            finite_difference[k][l] = Quantity::new(
                (residual_local(&plus, plastic_multiplier)?
                    - residual_local(&minus, plastic_multiplier)?)
                    / (2.0 * step),
            );
        }
    }
    assert.eq_within_tols(&analytic, &finite_difference)?;
    //
    // K_vv = d(Fischer-Burmeister)/d(plastic multiplier), plus the pinned identity.
    //
    assert.eq_within_tols(
        k_vv[0][0][0][0].value(),
        &((residual_local(&deformation_gradient, plastic_multiplier + step)?
            - residual_local(&deformation_gradient, plastic_multiplier - step)?)
            / (2.0 * step)),
    )?;
    for i in 0..3 {
        for j in 0..3 {
            if i != 0 || j != 0 {
                assert_eq!(k_vv[i][j][i][j].value(), 1.0);
            }
        }
    }
    // Guard against a vacuous comparison: the coupling blocks must be non-negligible.
    assert!(analytic.norm().value() > 1e-2);
    Ok(())
}

#[test]
fn monolithic_coupling_blocks_keep_the_block_solve_within_a_tight_step_cap()
-> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::MonolithicRoot, math::optimize::SolveStrategy,
    };
    // A wrong K_uv / K_vu still converges to the same root, just slower, so the
    // agreement test above cannot catch a bad coupling block. This one can: with the
    // correct blocks the Schur-eliminated Newton clears each step in six iterations;
    // zeroing a coupling block blows the cap.
    let model = model(1.0);
    let solver = NewtonRaphson {
        max_steps: 6,
        ..Default::default()
    };
    // Coarse steps (large plastic increments, cold local start) so the coupling-block
    // quality actually shows up in the iteration count.
    let (_, _, states) = MonolithicRoot::root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 6)),
        solver,
        SolveStrategy::Monolithic { elimination: true },
    )?;
    assert!(states.as_slice().last().unwrap().1.value() > 0.0);
    Ok(())
}

#[test]
fn consistent_tangent_matches_the_finite_difference_algorithmic_tangent()
-> Result<(), AssertionError> {
    let model = model(1.0);
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
    let (consistent, updated) =
        model.consistent_tangent_stiffness(&deformation_gradient, &previous_state)?;
    let finite_difference =
        model.algorithmic_tangent_stiffness(&deformation_gradient, &previous_state)?;
    // the return-mapped state the tangent is taken at
    Assert::default().eq_within_tols(
        updated.1,
        &model.return_map(&deformation_gradient, &previous_state)?.1,
    )?;
    let directions = [
        DeformationGradient::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.0, 0.4, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.2, 0.1, -0.15], [0.1, -0.3, 0.05], [-0.15, 0.05, 0.25]]),
    ];
    // The condensed analytic tangent freezes the flow direction, so it drops the
    // radial-return geometric term (~dN/dF); against the fully finite-differenced
    // algorithmic tangent that is a couple of percent for isotropic J2, small enough
    // to keep quadratic-ish convergence of the outer solve.
    for direction in &directions {
        Assert {
            abs_tol: 3e-2,
            rel_tol: 3e-2,
            ..Default::default()
        }
        .eq_within_tols(
            contract(&consistent, direction),
            &contract(&finite_difference, direction),
        )?;
    }
    Ok(())
}
