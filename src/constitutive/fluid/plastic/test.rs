use super::*;
use crate::{
    EPSILON,
    math::assert::{Assert, AssertionError, perturbation},
    math::{ContractWith, TensorArray},
};

fn model() -> PlasticFlow {
    PlasticFlow {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(1.0),
    }
}

fn deviatoric_mandel_stress() -> MandelStressElastic {
    MandelStressElastic::from([[1.3, 0.7, -0.4], [0.7, -0.9, 1.1], [-0.4, 1.1, -0.4]])
}

#[test]
fn flow_direction_has_unit_norm() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().flow_direction(&deviatoric_mandel_stress())?.norm(),
        &Quantity::new(1.0),
    )
}

#[test]
fn flow_direction_is_zero_at_zero_stress() -> Result<(), AssertionError> {
    Assert::zero(&model().flow_direction(&MandelStressElastic::zero())?)
}

#[test]
fn yield_function_reduces_to_yield_stress_offset() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    Assert::default().eq_within_tols(
        model.yield_function(&stress, Quantity::default())?,
        &(stress.norm() - model.initial_yield_stress()),
    )
}

#[test]
fn yield_function_hardens_with_equivalent_plastic_strain() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    Assert::default().eq_within_tols(
        model.yield_function(&stress, Quantity::new(1.0))?,
        &(model.yield_function(&stress, Quantity::default())? - model.hardening_slope()),
    )
}

#[test]
fn plastic_stretching_rate_is_multiplier_times_flow_direction() -> Result<(), AssertionError> {
    let model = model();
    let stress = deviatoric_mandel_stress();
    let multiplier = Rate::per_second(0.3);
    Assert::default().eq_within_tols(
        &model.plastic_stretching_rate(&stress, multiplier)?,
        &(model.flow_direction(&stress)? * multiplier),
    )
}

#[test]
fn deviatoric_mandel_stress_from_finite_difference_of_dissipation_potential()
-> Result<(), AssertionError> {
    let model = model();
    let flow_direction = model.flow_direction(&deviatoric_mandel_stress())?;
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate = flow_direction.clone() * Rate::per_second(0.3);
    let mandel_stress_on_surface = flow_direction * yield_stress;
    let mut finite_difference = MandelStressElastic::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = plastic_stretching_rate.clone();
            plus[i][j] += perturbation(0.5 * EPSILON);
            let mut minus = plastic_stretching_rate.clone();
            minus[i][j] -= perturbation(0.5 * EPSILON);
            finite_difference[i][j] = (model.dissipation_potential(plus, yield_stress)?
                - model.dissipation_potential(minus, yield_stress)?)
                / Quantity::<Rate>::new(EPSILON);
        }
    }
    Assert::default().eq_within_fd_tol(&mandel_stress_on_surface, &finite_difference)
}

fn voce() -> VoceFlow {
    VoceFlow {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(0.2),
        saturation_stress: Stress::pascals(1.5),
        saturation_rate: 8.0,
    }
}

#[test]
fn the_default_hardening_modulus_is_the_slope() -> Result<(), AssertionError> {
    let model = model();
    [0.0, 0.3, 2.0].into_iter().try_for_each(|strain| {
        Assert::default().eq_within_tols(
            model.hardening_modulus(Quantity::new(strain))?,
            &model.hardening_slope(),
        )
    })
}

#[test]
fn hardening_modulus_is_the_derivative_of_the_yield_stress() -> Result<(), AssertionError> {
    let (model, step) = (voce(), 1e-6);
    for strain in [0.0, 0.01, 0.05, 0.3, 1.0] {
        let finite_difference = (model.yield_stress(Quantity::new(strain + step))?
            - model.yield_stress(Quantity::new(strain - step))?)
            / (2.0 * step);
        let modulus = model.hardening_modulus(Quantity::new(strain))?;
        assert!(
            (modulus.value() - finite_difference.value()).abs()
                <= 1e-6 * (1.0 + modulus.value().abs()),
            "strain {strain}: modulus {} vs finite difference {}",
            modulus.value(),
            finite_difference.value(),
        );
    }
    Ok(())
}

#[test]
fn voce_hardening_starts_at_the_initial_yield_stress_and_saturates() -> Result<(), AssertionError> {
    let model = voce();
    Assert::default().eq_within_tols(
        model.yield_stress(Quantity::default())?,
        &model.initial_yield_stress(),
    )?;
    // the initial slope is H + Q b, and past saturation only the linear part H remains
    Assert::default().eq_within_tols(
        model.hardening_modulus(Quantity::default())?,
        &model.hardening_slope(),
    )?;
    Assert::default().eq_within_tols(
        model.hardening_slope(),
        &(Stress::pascals(0.2) + Stress::pascals(1.5) * 8.0),
    )?;
    Assert::default().eq_within_tols(
        model.hardening_modulus(Quantity::new(10.0))?,
        &Stress::pascals(0.2),
    )
}

#[test]
fn fenchel_equality() -> Result<(), AssertionError> {
    let model = model();
    let flow_direction = model.flow_direction(&deviatoric_mandel_stress())?;
    let yield_stress = model.initial_yield_stress();
    let plastic_stretching_rate = flow_direction.clone() * Rate::per_second(0.3);
    let mandel_stress_on_surface = flow_direction * yield_stress;
    Assert::default().eq_within_tols(
        model.dissipation_potential(plastic_stretching_rate.clone(), yield_stress)?,
        &ContractWith::contract_with(&mandel_stress_on_surface, &plastic_stretching_rate),
    )
}
