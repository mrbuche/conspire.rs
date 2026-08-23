use super::*;
use crate::{
    EPSILON,
    math::assert::{Assert, AssertionError, perturbation},
    math::TensorArray,
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
    MandelStressElastic::from([
        [1.3, 0.7, -0.4],
        [0.7, -0.9, 1.1],
        [-0.4, 1.1, -0.4],
    ])
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
    Assert::default().eq_within_fd_tol(&deviatoric_mandel_stress(), &finite_difference)
}
