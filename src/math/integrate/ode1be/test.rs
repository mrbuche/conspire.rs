use super::{
    super::{
        super::{
            optimize::{GradientDescent, NewtonRaphson, Optimization},
            test::TestError,
            Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1Vec, TensorRank2, Vector,
        },
        test::zero_to_tau,
    },
    Implicit, Ode1be,
};
use std::f64::consts::TAU;

const LENGTH: usize = 33;
const TOLERANCE: TensorRank0 = 1e6 * crate::ABS_TOL;

#[test]
fn do_2_error_cases_and_1_interp_case() {
    todo!()
}

macro_rules! test_ode1be {
    ($optimization: expr) => {
        #[test]
        fn first_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = Ode1be {
                optimization: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank0| -y,
                |_: &TensorRank0, _: &TensorRank0| -1.0,
                0.0,
                1.0,
                &[0.0, TAU],
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    ((-t).exp() - y).abs() < TOLERANCE || ((-t).exp() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn first_order_tensor_rank_0_eval_times() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = Ode1be {
                optimization: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank0| -y,
                |_: &TensorRank0, _: &TensorRank0| -1.0,
                0.0,
                1.0,
                &zero_to_tau::<LENGTH>(),
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    ((-t).exp() - y).abs() < TOLERANCE || ((-t).exp() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn second_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<2, 1>) = Ode1be {
                optimization: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<2, 1>| TensorRank1::new([y[1], -y[0]]),
                |_: &TensorRank0, _: &TensorRank1<2, 1>| {
                    TensorRank2::new([[0.0, -1.0], [1.0, 0.0]])
                },
                0.0,
                TensorRank1::new([0.0, 1.0]),
                &[0.0, TAU],
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn third_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<3, 1>) = Ode1be {
                optimization: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<3, 1>| TensorRank1::new([y[1], y[2], -y[1]]),
                |_: &TensorRank0, _: &TensorRank1<3, 1>| {
                    TensorRank2::new([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
                },
                0.0,
                TensorRank1::new([0.0, 1.0, 0.0]),
                &[0.0, TAU],
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn fourth_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<4, 1>) = Ode1be {
                optimization: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<4, 1>| TensorRank1::new([y[1], y[2], y[3], y[0]]),
                |_: &TensorRank0, _: &TensorRank1<4, 1>| {
                    TensorRank2::new([
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ])
                },
                0.0,
                TensorRank1::new([0.0, 1.0, 0.0, -1.0]),
                &[0.0, TAU],
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
    };
}

mod gradient_descent {
    use super::*;
    test_ode1be!(Optimization::GradientDescent(GradientDescent {
        ..Default::default()
    }));
}

mod newton_raphson {
    use super::*;
    test_ode1be!(Optimization::NewtonRaphson(NewtonRaphson {
        check_minimum: false,
        ..Default::default()
    }));
}
