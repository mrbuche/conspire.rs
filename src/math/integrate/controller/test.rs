use super::StepController;

const CONTROLLER: StepController = StepController::new(0.9, 2.0, 0.2, 5.0);

#[test]
fn zero_error_grows_to_the_ceiling() {
    assert_eq!(CONTROLLER.scale(0.0, 1e-6), 5.0);
}

#[test]
fn matches_the_law_inside_the_clamp() {
    let scale = CONTROLLER.scale(4e-6, 1e-6);
    assert!((scale - 0.9 * (0.25_f64).powf(0.5)).abs() < 1e-15);
}

#[test]
fn error_at_the_tolerance_scales_by_beta() {
    assert!((CONTROLLER.scale(1e-6, 1e-6) - 0.9).abs() < 1e-15);
}

#[test]
fn clamps_to_the_cut_and_the_ceiling() {
    assert_eq!(CONTROLLER.scale(1e6, 1e-6), 0.2);
    assert_eq!(CONTROLLER.scale(1e-30, 1e-6), 5.0);
}

#[test]
fn a_higher_order_grows_the_step_less_for_the_same_margin() {
    let low = StepController::new(0.9, 2.0, 0.2, 5.0).scale(1e-9, 1e-6);
    let high = StepController::new(0.9, 8.0, 0.2, 5.0).scale(1e-9, 1e-6);
    assert!(high < low);
}
