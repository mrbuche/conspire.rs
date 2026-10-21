use super::StepController;
use crate::{
    math::Quantity,
    units::{Dimensionless, Stress},
};

const CONTROLLER: StepController = StepController::new(0.9, 2.0, 0.2, 5.0);

fn scale(controller: &StepController, error: f64, tolerance: f64) -> f64 {
    controller.scale(
        Quantity::<Dimensionless>::new(error),
        Quantity::new(tolerance),
    )
}

#[test]
fn zero_error_grows_to_the_ceiling() {
    assert_eq!(scale(&CONTROLLER, 0.0, 1e-6), 5.0);
}

#[test]
fn matches_the_law_inside_the_clamp() {
    let scale = scale(&CONTROLLER, 4e-6, 1e-6);
    assert!((scale - 0.9 * (0.25_f64).powf(0.5)).abs() < 1e-15);
}

#[test]
fn error_at_the_tolerance_scales_by_beta() {
    assert!((scale(&CONTROLLER, 1e-6, 1e-6) - 0.9).abs() < 1e-15);
}

#[test]
fn clamps_to_the_cut_and_the_ceiling() {
    assert_eq!(scale(&CONTROLLER, 1e6, 1e-6), 0.2);
    assert_eq!(scale(&CONTROLLER, 1e-30, 1e-6), 5.0);
}

#[test]
fn a_higher_order_grows_the_step_less_for_the_same_margin() {
    let low = scale(&StepController::new(0.9, 2.0, 0.2, 5.0), 1e-9, 1e-6);
    let high = scale(&StepController::new(0.9, 8.0, 0.2, 5.0), 1e-9, 1e-6);
    assert!(high < low);
}

#[test]
fn holds_a_dimensional_error_to_a_tolerance_of_the_same_unit() {
    let scale = CONTROLLER.scale(Quantity::<Stress>::new(4e-6), Quantity::new(1e-6));
    assert!((scale - 0.9 * (0.25_f64).powf(0.5)).abs() < 1e-15);
}
