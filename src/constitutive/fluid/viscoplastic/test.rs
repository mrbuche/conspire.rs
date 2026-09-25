use super::*;
use crate::{
    EPSILON,
    math::assert::{Assert, AssertionError, perturbation},
    math::{ContractWith, TensorArray},
};

fn model() -> ViscoplasticFlow {
    ViscoplasticFlow {
        yield_stress: Stress::pascals(2.0),
        hardening_slope: Stress::pascals(1.0),
        rate_sensitivity: 0.25,
        reference_flow_rate: Rate::per_second(0.1),
    }
}

fn deviatoric_mandel_stress() -> MandelStressElastic {
    MandelStressElastic::from([[1.3, 0.7, -0.4], [0.7, -0.9, 1.1], [-0.4, 1.1, -0.4]])
}

#[test]
fn plastic_stretching_rate_from_finite_difference_of_dual_dissipation_potential()
-> Result<(), AssertionError> {
    let model = model();
    let mandel_stress = deviatoric_mandel_stress();
    let yield_stress = model.yield_stress;
    let mut finite_difference = StretchingRatePlastic::zero();
    for i in 0..3 {
        for j in 0..3 {
            let mut plus = deviatoric_mandel_stress();
            plus[i][j] += perturbation(0.5 * EPSILON);
            let mut minus = deviatoric_mandel_stress();
            minus[i][j] -= perturbation(0.5 * EPSILON);
            finite_difference[i][j] = (model.dual_dissipation_potential(plus, yield_stress)?
                - model.dual_dissipation_potential(minus, yield_stress)?)
                / Quantity::<Stress>::new(EPSILON);
        }
    }
    Assert::default().eq_within_fd_tol(
        &model.plastic_stretching_rate(mandel_stress, yield_stress)?,
        &finite_difference,
    )
}

#[test]
fn deviatoric_mandel_stress_from_finite_difference_of_dissipation_potential()
-> Result<(), AssertionError> {
    let model = model();
    let yield_stress = model.yield_stress;
    let plastic_stretching_rate =
        model.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?;
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
    Assert::default().eq_within_fd_tol(deviatoric_mandel_stress(), &finite_difference)
}

#[test]
fn fenchel_equality() -> Result<(), AssertionError> {
    let model = model();
    let yield_stress = model.yield_stress;
    let plastic_stretching_rate =
        model.plastic_stretching_rate(deviatoric_mandel_stress(), yield_stress)?;
    Assert::default().eq_within_tols(
        model.dissipation_potential(plastic_stretching_rate.clone(), yield_stress)?
            + model.dual_dissipation_potential(deviatoric_mandel_stress(), yield_stress)?,
        &ContractWith::contract_with(&deviatoric_mandel_stress(), &plastic_stretching_rate),
    )
}

#[test]
fn plastic_stretching_rate_tangent_matches_finite_difference() -> Result<(), AssertionError> {
    let model = model();
    let yield_stress = model.yield_stress;
    let tangent =
        model.plastic_stretching_rate_tangent(&deviatoric_mandel_stress(), yield_stress)?;
    let mut finite_difference = StretchingRatePlasticTangent::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut plus = deviatoric_mandel_stress();
            plus[k][l] += perturbation(0.5 * EPSILON);
            let rate_plus = model.plastic_stretching_rate(plus, yield_stress)?;
            let mut minus = deviatoric_mandel_stress();
            minus[k][l] -= perturbation(0.5 * EPSILON);
            let rate_minus = model.plastic_stretching_rate(minus, yield_stress)?;
            for i in 0..3 {
                for j in 0..3 {
                    finite_difference[i][j][k][l] =
                        (rate_plus[i][j] - rate_minus[i][j]) / Quantity::<Stress>::new(EPSILON);
                }
            }
        }
    }
    Assert::default().eq_within_fd_tol(&tangent, &finite_difference)
}

#[test]
fn plastic_stretching_rate_tangent_yield_matches_finite_difference() -> Result<(), AssertionError> {
    let model = model();
    let yield_stress = model.yield_stress;
    let tangent =
        model.plastic_stretching_rate_tangent_yield(deviatoric_mandel_stress(), yield_stress)?;
    let rate_plus = model.plastic_stretching_rate(
        deviatoric_mandel_stress(),
        yield_stress + Quantity::new(0.5 * EPSILON),
    )?;
    let rate_minus = model.plastic_stretching_rate(
        deviatoric_mandel_stress(),
        yield_stress - Quantity::new(0.5 * EPSILON),
    )?;
    Assert::default().eq_within_fd_tol(
        &tangent,
        &((rate_plus - rate_minus) / Quantity::<Stress>::new(EPSILON)),
    )
}

#[test]
fn yield_stress_slope_is_the_hardening_slope() -> Result<(), AssertionError> {
    let model = model();
    let plus = model.yield_stress(Quantity::new(0.5 * EPSILON))?;
    let minus = model.yield_stress(Quantity::new(-0.5 * EPSILON))?;
    Assert::default().eq_within_fd_tol(model.hardening_slope(), &((plus - minus) / EPSILON))
}
