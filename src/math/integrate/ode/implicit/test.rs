macro_rules! test_implicit_fixed_step {
    ($integration: expr) => {
        use crate::units::{Rate, Time};
        use crate::{
            EPSILON,
            math::{
                Quantity, Scalar, Tensor, TensorVector,
                assert::AssertionError,
                integrate::{FixedStep, IntegrationError, Times},
            },
        };
        const TIME_STEP: Quantity<Time> = Quantity::new(0.1);
        const TOLERANCE: Scalar = 0.1;
        /// The rate the dimensionless fixtures are stated against, there being
        /// no rate to divide a state by otherwise.
        const RATE: Quantity<Rate> = Quantity::new(1.0);
        /// The state and the rate the solution is reported as.
        type States = TensorVector<Quantity>;
        type Rates = TensorVector<Quantity<Rate>>;
        #[test]
        fn finite_difference() -> Result<(), AssertionError> {
            use crate::math::integrate::{ImplicitFirstOrder, ImplicitZerothOrder};
            let t = Quantity::<Time>::new(0.55);
            let y = (t * RATE).sin();
            let function = |t: Quantity<Time>, y: &Quantity| {
                Ok::<Quantity<Rate>, IntegrationError>(*y * *y * (t * RATE) * RATE * -0.5)
            };
            let jacobian = |t: Quantity<Time>, y: &Quantity| {
                Ok::<Quantity<Rate>, IntegrationError>(*y * (t * RATE) * RATE * -1.0)
            };
            let dt = TIME_STEP;
            let t_trial = t + dt;
            let y_trial = y + function(t, &y)? * dt;
            let finite_difference =
                (ImplicitZerothOrder::<Quantity, States, Rates>::residual(
                    &$integration,
                    &function,
                    t,
                    &y,
                    t_trial,
                    &(y_trial + 0.5 * EPSILON),
                    dt,
                )? - ImplicitZerothOrder::<Quantity, States, Rates>::residual(
                    &$integration,
                    &function,
                    t,
                    &y,
                    t_trial,
                    &(y_trial - 0.5 * EPSILON),
                    dt,
                )?) / EPSILON;
            $crate::math::assert::Assert::default().eq_within_fd_tol(
                &ImplicitFirstOrder::<Quantity, Quantity, States, Rates>::hessian(
                    &$integration,
                    &jacobian,
                    t,
                    &y,
                    t_trial,
                    &y_trial,
                    dt,
                )?,
                &finite_difference,
            )
        }
        mod gradient_descent {
            use super::*;
            use crate::math::{integrate::ImplicitZerothOrder, optimize::GradientDescent};
            // A scalar unknown is a `Quantity`, a bare scalar carrying no unit
            // for the solver's step size to be measured against.
            #[test]
            fn first_order_tensor_rank_0() -> Result<(), AssertionError> {
                $crate::math::assert::Assert::eq(
                    &FixedStep::<Time>::dt(&$integration),
                    &TIME_STEP,
                )?;
                let (time, solution, function): (Times, States, Rates) = $integration.integrate(
                    |t: Quantity<Time>, _: &Quantity| Ok((t * RATE) * RATE),
                    &[Quantity::new(0.0), Quantity::new(1.0)],
                    Quantity::new(0.0),
                    GradientDescent::default(),
                )?;
                time.iter()
                    .zip(solution.iter().zip(function.iter()))
                    .for_each(|(t, (y, f))| {
                        let t = (*t * RATE).value();
                        assert!(
                            (0.5 * t * t - y.value()).abs() < TOLERANCE
                                && (t - f.value()).abs() < TOLERANCE
                        )
                    });
                Ok(())
            }
        }
        mod newton_raphson {
            use super::*;
            use crate::math::{integrate::ImplicitFirstOrder, optimize::NewtonRaphson};
            #[test]
            fn first_order_tensor_rank_0() -> Result<(), AssertionError> {
                $crate::math::assert::Assert::eq(
                    &FixedStep::<Time>::dt(&$integration),
                    &TIME_STEP,
                )?;
                // The Jacobian of the residual is named, a derivative telling
                // which tensor it came from only in the one direction.
                let (time, solution, function): (Times, States, Rates) =
                    ImplicitFirstOrder::<Quantity, Quantity, States, Rates>::integrate(
                        &$integration,
                        |t: Quantity<Time>, _: &Quantity| Ok((t * RATE) * RATE),
                        |_: Quantity<Time>, _: &Quantity| Ok(RATE),
                        &[Quantity::new(0.0), Quantity::new(1.0)],
                        Quantity::new(0.0),
                        NewtonRaphson::default(),
                    )?;
                time.iter()
                    .zip(solution.iter().zip(function.iter()))
                    .for_each(|(t, (y, f))| {
                        let t = (*t * RATE).value();
                        assert!(
                            (0.5 * t * t - y.value()).abs() < TOLERANCE
                                && (t - f.value()).abs() < TOLERANCE
                        )
                    });
                Ok(())
            }
        }
    };
}
pub(crate) use test_implicit_fixed_step;
