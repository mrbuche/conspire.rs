use super::{Armijo, LineSearchError, ObjectiveMerit, Wolfe};
use crate::{math::Quantity, units::Dimensionless};

fn search(strong: bool) -> f64 {
    Wolfe {
        control_1: 1e-3,
        control_2: 4e-1,
        cut_back: 5e-1,
        max_steps: 100,
        strong,
    }
    .search(
        |n: Quantity<Dimensionless>| Ok((Quantity::new(1.0) - n).powi(2) / 2.0),
        |n: Quantity<Dimensionless>| Ok(Quantity::<Dimensionless>::new(1.0) - n),
        Quantity::new(3.0),
    )
    .unwrap()
    .value()
}

#[test]
fn wolfe_curvature_condition_follows_the_flag() {
    assert_eq!(search(false), 1.5);
    assert_eq!(search(true), 0.75)
}

mod vetting {
    use super::*;
    fn armijo(
        value: Result<f64, String>,
        slope: f64,
        step: f64,
    ) -> Result<Quantity<Dimensionless>, LineSearchError> {
        Armijo::default().search(
            |n: Quantity<Dimensionless>| {
                value.clone().map(|value| {
                    Quantity::<Dimensionless>::new(value)
                        - n * Quantity::<Dimensionless>::new(slope)
                })
            },
            |_: Quantity<Dimensionless>| Ok(Quantity::<Dimensionless>::new(slope)),
            Quantity::new(step),
        )
    }
    #[test]
    fn a_descent_is_searched() {
        assert!(armijo(Ok(1.0), 1.0, 1.0).is_ok())
    }
    #[test]
    fn a_step_of_no_size_is_refused() {
        assert!(matches!(
            armijo(Ok(1.0), 1.0, 0.0),
            Err(LineSearchError::NegativeStepSize(..))
        ))
    }
    #[test]
    fn a_negative_step_is_refused() {
        assert!(matches!(
            armijo(Ok(1.0), 1.0, -1.0),
            Err(LineSearchError::NegativeStepSize(..))
        ))
    }
    #[test]
    fn a_flat_direction_is_not_a_descent() {
        assert!(matches!(
            armijo(Ok(1.0), 0.0, 1.0),
            Err(LineSearchError::NotDescentDirection(..))
        ))
    }
    #[test]
    fn a_rising_direction_is_not_a_descent() {
        assert!(matches!(
            armijo(Ok(1.0), -1.0, 1.0),
            Err(LineSearchError::NotDescentDirection(..))
        ))
    }
    #[test]
    fn a_start_that_cannot_be_evaluated_is_invalid() {
        assert!(matches!(
            armijo(Err("no".to_string()), 1.0, 1.0),
            Err(LineSearchError::InvalidStartingPoint(..))
        ))
    }
}
