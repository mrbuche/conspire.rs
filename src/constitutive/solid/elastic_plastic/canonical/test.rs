use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{
            Plastic, PlasticFlow, PlasticStateVariables, RateIndependentPlastic, VoceFlow,
        },
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
        optimize::{NewtonRaphson, SolveStrategy},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic, FirstPiolaKirchhoffStress},
    units::{Stress, Time},
};

/// `FirstOrderRoot::root` with the default (condensed) strategy, for tests that
/// don't care which strategy is used.
fn root(
    model: &Canonical<NeoHookean, PlasticFlow>,
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
    FirstOrderRoot::root(
        model,
        applied_load,
        solver,
        SolveStrategy::Condensed(NewtonRaphson::default()),
    )
}

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

// saturates within a few percent plastic strain, so the hardening modulus changes a lot
// over a step
fn voce_model() -> Canonical<NeoHookean, VoceFlow> {
    Canonical::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        VoceFlow {
            yield_stress: Stress::pascals(2.0),
            hardening_slope: Stress::pascals(0.2),
            saturation_stress: Stress::pascals(1.5),
            saturation_rate: 8.0,
        },
    ))
}

#[test]
fn a_composed_model_forwards_the_hardening_law() -> Result<(), AssertionError> {
    // a wrapper that forwarded only the initial yield stress and slope would silently
    // fall back to the linear default
    let model = voce_model();
    for strain in [0.0, 0.01, 0.05, 0.4] {
        let strain = Quantity::new(strain);
        Assert::default()
            .eq_within_tols(model.yield_stress(strain)?, &model.1.yield_stress(strain)?)?;
        Assert::default().eq_within_tols(
            model.hardening_modulus(strain)?,
            &model.1.hardening_modulus(strain)?,
        )?;
    }
    assert!(
        (model.hardening_modulus(Quantity::new(0.4))? - model.hardening_slope())
            .value()
            .abs()
            > 1e-3,
        "the modulus must vary with strain for this test to mean anything"
    );
    Ok(())
}

#[test]
fn return_map_satisfies_the_implicit_step_with_nonlinear_hardening() -> Result<(), AssertionError> {
    let model = voce_model();
    let first = DeformationGradient::from([[1.5, 0.35, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.15]]);
    let second = DeformationGradient::from([[1.55, 0.5, 0.1], [0.2, 0.95, 0.3], [-0.1, 0.05, 1.1]]);
    let state = assert_implicit_step(&model, &first, &model.initial_state())?;
    assert_implicit_step(&model, &second, &state)?;
    Ok(())
}

#[test]
fn condensed_matches_the_return_map() -> Result<(), AssertionError> {
    assert_condensed_matches(&model(1.0))?;
    assert_condensed_matches(&voce_model())
}

fn assert_condensed_matches<M: ElasticPlastic>(model: &M) -> Result<(), AssertionError> {
    let steps = [
        DeformationGradient::from([[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        DeformationGradient::from([[1.5, 0.35, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.15]]),
        DeformationGradient::from([[1.55, 0.5, 0.1], [0.2, 0.95, 0.3], [-0.1, 0.05, 1.1]]),
    ];
    let assert = Assert {
        abs_tol: 1e-8,
        rel_tol: 1e-8,
        ..Default::default()
    };
    let mut state = model.initial_state();
    for deformation_gradient in &steps {
        let (stress, _, updated) = model.condensed(deformation_gradient, &state)?;
        let reference_state = model.return_map(deformation_gradient, &state)?;
        assert.eq_within_tols(&updated.0, &reference_state.0)?;
        assert.eq_within_tols(updated.1, &reference_state.1)?;
        assert.eq_within_tols(
            &stress,
            &model.first_piola_kirchhoff_stress(deformation_gradient, &reference_state.0)?,
        )?;
        state = updated
    }
    assert!(state.1.value() > 0.0);
    Ok(())
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
    let (_, _, states) = root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(0.02, 4)),
        solver(),
    )?;
    states.as_slice().iter().try_for_each(|state| {
        assert_eq!(state.1.value(), 0.0);
        Assert::default().eq_within_tols(&state.0, &DeformationGradientPlastic::identity())
    })
}

#[test]
fn yields_and_returns_to_the_surface() -> Result<(), AssertionError> {
    let model = model(1.0);
    let (_, deformation_gradients, states) = root(
        &model,
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
        let (_, deformation_gradients, states) = root(
            &model,
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
    let (_, deformation_gradients, states) = root(
        &model,
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
fn consistent_tangent_keeps_the_outer_solve_within_a_tight_step_cap() -> Result<(), AssertionError>
{
    // Coarse steps, so a degraded tangent shows up in the iteration count: three
    // iterations per step is not enough even with the exact tangent.
    let model = model(1.0);
    let solver = NewtonRaphson {
        max_steps: 4,
        ..Default::default()
    };
    let (_, _, states) = root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 6)),
        solver,
    )?;
    assert!(states.as_slice().last().unwrap().1.value() > 0.0);
    Ok(())
}

#[test]
fn monolithic_strategies_agree_with_each_other() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::FirstOrderRoot, math::optimize::SolveStrategy,
    };
    let model = model(1.0);
    let steps = times(0.5, 40);
    // Condensed is the reference: it converges the local block before every outer step.
    let (_, reference_gradients, reference_states) = FirstOrderRoot::root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &steps),
        NewtonRaphson::default(),
        SolveStrategy::Condensed(NewtonRaphson::default()),
    )?;
    let reference_gradient = reference_gradients.as_slice().last().unwrap();
    let reference_strain = reference_states.as_slice().last().unwrap().1;
    assert!(reference_strain.value() > 0.0);
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
    ] {
        let (_, gradients, states) = FirstOrderRoot::root(
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
fn monolithic_coupling_blocks_keep_the_block_solve_within_a_tight_step_cap()
-> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::FirstOrderRoot, math::optimize::SolveStrategy,
    };
    // A wrong K_uv / K_vu still converges to the same root, just slower, so the
    // agreement test above cannot catch a bad coupling block. This one can: with the
    // correct blocks the Schur-eliminated Newton clears each step in six iterations;
    // zeroing a coupling block blows the cap. The dN/dF term does not change the count
    // on this proportional path (the flow direction barely moves along it) -- the
    // finite-difference test above is what guards that term.
    let model = model(1.0);
    let solver = NewtonRaphson {
        max_steps: 6,
        ..Default::default()
    };
    // Coarse steps (large plastic increments, cold local start) so the coupling-block
    // quality actually shows up in the iteration count.
    let (_, _, states) = FirstOrderRoot::root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 6)),
        solver,
        SolveStrategy::Monolithic { elimination: true },
    )?;
    assert!(states.as_slice().last().unwrap().1.value() > 0.0);
    Ok(())
}

#[test]
fn consistent_tangent_matches_the_finite_difference_through_the_return_map()
-> Result<(), AssertionError> {
    let model = model(1.0);
    let (_, deformation_gradients, states) = root(
        &model,
        AppliedLoad::UniaxialStress(ramp, &times(0.5, 100)),
        solver(),
    )?;
    let deformation_gradient = deformation_gradients.as_slice()[90].clone();
    let previous_state = states.as_slice()[89].clone();
    assert!(
        states.as_slice()[90].1.value() > 0.0,
        "step 90 must be plastic"
    );
    let (_, consistent, updated) = model.condensed(&deformation_gradient, &previous_state)?;
    let continuum =
        model.first_piola_kirchhoff_tangent_stiffness(&deformation_gradient, &updated.0)?;
    // the return-mapped state the tangent is taken at
    Assert::default().eq_within_tols(
        updated.1,
        &model.return_map(&deformation_gradient, &previous_state)?.1,
    )?;
    let step = 1.0e-6;
    let directions = [
        DeformationGradient::from([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.0, 0.4, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        DeformationGradient::from([[0.2, 0.1, -0.15], [0.1, -0.3, 0.05], [-0.15, 0.05, 0.25]]),
    ];
    // The analytic tangent carries the radial-return dN/dF term, so it is exact and
    // only the finite-difference truncation separates it from the difference quotient.
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
        .eq_within_tols(contract(&consistent, direction), &finite_difference)?;
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

fn assert_implicit_step<M: ElasticPlastic>(
    model: &M,
    deformation_gradient: &DeformationGradient,
    previous_state: &PlasticStateVariables,
) -> Result<PlasticStateVariables, AssertionError> {
    let (previous_f_p, &previous_strain): (&DeformationGradientPlastic, &Quantity) =
        previous_state.into();
    let updated_state = model.return_map(deformation_gradient, previous_state)?;
    let (f_p, &strain): (&DeformationGradientPlastic, &Quantity) = (&updated_state).into();
    let plastic_multiplier = (strain - previous_strain).value();
    assert!(plastic_multiplier > 0.0, "the step must be plastic");
    let deviatoric = model.mandel_stress(deformation_gradient, f_p)?.deviatoric();
    // the yield condition holds at the end of the step
    Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..Default::default()
    }
    .eq_within_tols(model.yield_function(&deviatoric, strain)?.value(), &0.0)?;
    // and the flow direction is the end-of-step one, not the trial one
    let direction = {
        let direction = model.flow_direction(&deviatoric)?;
        (&direction + direction.transpose()) * 0.5
    };
    let implicit_f_p = (&direction * plastic_multiplier).expm().unwrap() * previous_f_p;
    Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..Default::default()
    }
    .eq_within_tols(f_p, &implicit_f_p)?;
    Ok(updated_state)
}

#[test]
fn return_map_is_the_fully_implicit_step_when_the_loading_is_not_proportional()
-> Result<(), AssertionError> {
    let model = model(1.0);
    let first = DeformationGradient::from([[1.5, 0.35, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.15]]);
    let second = DeformationGradient::from([[1.55, 0.5, 0.1], [0.2, 0.95, 0.3], [-0.1, 0.05, 1.1]]);
    let state = assert_implicit_step(&model, &first, &model.initial_state())?;
    assert_implicit_step(&model, &second, &state)?;
    Ok(())
}

#[test]
fn return_map_solves_a_step_too_large_for_a_frozen_flow_direction() -> Result<(), AssertionError> {
    let model = model(1.0);
    let large = DeformationGradient::from([[2.6, 1.12, 0.32], [0.0, 0.68, 0.64], [0.0, 0.0, 1.48]]);
    assert_implicit_step(&model, &large, &model.initial_state())?;
    Ok(())
}

#[test]
fn monolithic_blocks_match_finite_difference_at_a_plastic_state() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::coupled::{
            SIZE, monolithic_plastic, monolithic_residual_local, monolithic_tangents,
        },
        math::Vector,
    };
    let model = model(1.0);
    let first = DeformationGradient::from([[1.5, 0.35, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.15]]);
    let previous_state = model.return_map(&first, &model.initial_state())?;
    assert!(
        previous_state.1.value() > 0.0,
        "the previous step must be plastic"
    );
    let deformation_gradient =
        DeformationGradient::from([[1.55, 0.5, 0.1], [0.2, 0.95, 0.3], [-0.1, 0.05, 1.1]]);
    // a symmetric, trace-free plastic increment and a multiplier that puts the trial
    // state off the yield surface, so the complementarity function is smooth here
    let mut local = Vector::zero(SIZE);
    [0.02, 0.01, 0.005, 0.01, -0.03, 0.0, 0.005, 0.0, 0.01, 0.04]
        .iter()
        .enumerate()
        .for_each(|(index, value)| local[index] = *value);
    let residual_global = |gradient: &DeformationGradient, local: &Vector| {
        Ok::<_, AssertionError>(model.first_piola_kirchhoff_stress(
            gradient,
            &monolithic_plastic(&model, &previous_state, local)?,
        )?)
    };
    let residual_local = |gradient: &DeformationGradient, local: &Vector| {
        Ok::<_, AssertionError>(monolithic_residual_local(
            &model,
            gradient,
            &previous_state,
            local,
        )?)
    };
    let (k_uu, k_vu, k_uv, k_vv) =
        monolithic_tangents(&model, &deformation_gradient, &previous_state, &local)?;
    let step = 1.0e-6;
    let close = |analytic: f64, finite_difference: f64, what: &str| {
        assert!(
            (analytic - finite_difference).abs() <= 1e-6 * (1.0 + analytic.abs()),
            "{what}: analytic {analytic} vs finite difference {finite_difference}",
        )
    };
    let perturbed = |k: usize, l: usize, sign: f64| {
        let mut gradient = deformation_gradient.clone();
        gradient[k][l] += Quantity::new(sign * step);
        gradient
    };
    let shifted = |c: usize, sign: f64| {
        let mut shifted = local.clone();
        shifted[c] += sign * step;
        shifted
    };
    let (mut coupling_u, mut coupling_v) = (0.0_f64, 0.0_f64);
    for k in 0..3 {
        for l in 0..3 {
            let d_stress = (residual_global(&perturbed(k, l, 1.0), &local)?
                - residual_global(&perturbed(k, l, -1.0), &local)?)
                / (2.0 * step);
            for i in 0..3 {
                for j in 0..3 {
                    close(
                        k_uu[i][j][k][l].value(),
                        d_stress[i][j].value(),
                        &format!("K_uu[{i}][{j}][{k}][{l}]"),
                    );
                }
            }
            let d_local = (residual_local(&perturbed(k, l, 1.0), &local)?
                - residual_local(&perturbed(k, l, -1.0), &local)?)
                / (2.0 * step);
            for row in 0..SIZE {
                close(
                    k_vu[row][3 * k + l],
                    d_local[row],
                    &format!("K_vu[{row}][{}]", 3 * k + l),
                );
                coupling_v = coupling_v.max(d_local[row].abs());
            }
        }
    }
    for column in 0..SIZE {
        let d_stress = (residual_global(&deformation_gradient, &shifted(column, 1.0))?
            - residual_global(&deformation_gradient, &shifted(column, -1.0))?)
            / (2.0 * step);
        for i in 0..3 {
            for j in 0..3 {
                close(
                    k_uv[3 * i + j][column],
                    d_stress[i][j].value(),
                    &format!("K_uv[{}][{column}]", 3 * i + j),
                );
                coupling_u = coupling_u.max(d_stress[i][j].value().abs());
            }
        }
        let d_local = (residual_local(&deformation_gradient, &shifted(column, 1.0))?
            - residual_local(&deformation_gradient, &shifted(column, -1.0))?)
            / (2.0 * step);
        for row in 0..SIZE {
            close(
                k_vv[row][column],
                d_local[row],
                &format!("K_vv[{row}][{column}]"),
            );
        }
    }
    // guard against a vacuous comparison: the coupling blocks must be non-negligible
    assert!(coupling_u > 1e-2 && coupling_v > 1e-2);
    Ok(())
}

#[test]
fn monolithic_strategies_agree_when_the_loading_is_not_proportional() -> Result<(), AssertionError>
{
    assert_strategies_agree_under_biaxial_loading(&model(1.0))
}

#[test]
fn monolithic_strategies_agree_with_nonlinear_hardening() -> Result<(), AssertionError> {
    assert_strategies_agree_under_biaxial_loading(&voce_model())
}

fn assert_strategies_agree_under_biaxial_loading<M: ElasticPlastic>(
    model: &M,
) -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_plastic::FirstOrderRoot, math::optimize::SolveStrategy,
    };
    let steps = times(0.5, 40);
    // F_11 and F_22 follow different histories, so the plastic flow direction rotates
    // and a direction frozen at the start of the step would no longer be the converged
    // step's own.
    let load = || {
        AppliedLoad::BiaxialStress(
            ramp,
            |t: Quantity<Time>| 1.0 + 0.6 * t.value() * t.value() - 0.2 * t.value(),
            &steps,
        )
    };
    let (_, reference_gradients, reference_states) = FirstOrderRoot::root(
        model,
        load(),
        NewtonRaphson::default(),
        SolveStrategy::Condensed(NewtonRaphson::default()),
    )?;
    let reference_gradient = reference_gradients.as_slice().last().unwrap();
    let reference_state = reference_states.as_slice().last().unwrap();
    assert!(reference_state.1.value() > 0.0);
    for strategy in [
        SolveStrategy::Monolithic { elimination: false },
        SolveStrategy::Monolithic { elimination: true },
    ] {
        let (_, gradients, states) =
            FirstOrderRoot::root(model, load(), NewtonRaphson::default(), strategy)?;
        let assert = Assert {
            abs_tol: 1e-9,
            rel_tol: 1e-9,
            ..Default::default()
        };
        assert.eq_within_tols(gradients.as_slice().last().unwrap(), reference_gradient)?;
        let state = states.as_slice().last().unwrap();
        assert.eq_within_tols(state.1, &reference_state.1)?;
        assert.eq_within_tols(&state.0, &reference_state.0)?;
    }
    Ok(())
}
