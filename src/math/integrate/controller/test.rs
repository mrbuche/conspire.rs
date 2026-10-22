use super::StepController;
use crate::{
    math::Quantity,
    units::{Dimensionless, Stress},
};

const CONTROLLER: StepController = StepController::new(0.9, 2.0, 0.2, 5.0);

fn scale(controller: &StepController, error: f64, tolerance: f64) -> Quantity {
    controller.scale(Dimensionless::of(error), Dimensionless::of(tolerance))
}

fn close(quantity: Quantity, expected: f64) -> bool {
    (quantity - Dimensionless::of(expected)).abs() < Dimensionless::of(1e-15)
}

#[test]
fn zero_error_grows_to_the_ceiling() {
    assert_eq!(scale(&CONTROLLER, 0.0, 1e-6), Dimensionless::of(5.0));
}

#[test]
fn matches_the_law_inside_the_clamp() {
    assert!(close(
        scale(&CONTROLLER, 4e-6, 1e-6),
        0.9 * (0.25_f64).powf(0.5)
    ));
}

#[test]
fn error_at_the_tolerance_scales_by_beta() {
    assert!(close(scale(&CONTROLLER, 1e-6, 1e-6), 0.9));
}

#[test]
fn clamps_to_the_cut_and_the_ceiling() {
    assert_eq!(scale(&CONTROLLER, 1e6, 1e-6), Dimensionless::of(0.2));
    assert_eq!(scale(&CONTROLLER, 1e-30, 1e-6), Dimensionless::of(5.0));
}

#[test]
fn a_higher_order_grows_the_step_less_for_the_same_margin() {
    let low = scale(&StepController::new(0.9, 2.0, 0.2, 5.0), 1e-9, 1e-6);
    let high = scale(&StepController::new(0.9, 8.0, 0.2, 5.0), 1e-9, 1e-6);
    assert!(high < low);
}

#[test]
fn holds_a_dimensional_error_to_a_tolerance_of_the_same_unit() {
    let scale = CONTROLLER.scale(Stress::pascals(4e-6), Stress::pascals(1e-6));
    assert!(close(scale, 0.9 * (0.25_f64).powf(0.5)));
}

#[test]
fn scaling_a_step_keeps_its_unit() {
    let mut step = Stress::pascals(2.0);
    step *= scale(&CONTROLLER, 1e-6, 1e-6);
    assert!((step - Stress::pascals(1.8)).abs() < Stress::pascals(1e-15));
}
