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
const TOLERANCE: TensorRank0 = 1e-5;

macro_rules! test_ode1be {
    ($optimization: expr) => {
        #[test]
        #[should_panic(expected = "The time must contain at least two entries.")]
        fn initial_time_not_less_than_final_time() {
            let _: (Vector, Vector) = Ode1be {
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank0| -y,
                |_: &TensorRank0, _: &TensorRank0| -1.0,
                0.0,
                1.0,
                &[0.0],
            )
            .unwrap();
        }
        #[test]
        #[should_panic(expected = "The initial time must precede the final time.")]
        fn length_time_less_than_two() {
            let _: (Vector, Vector) = Ode1be {
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank0| -y,
                |_: &TensorRank0, _: &TensorRank0| -1.0,
                0.0,
                1.0,
                &[0.0, 1.0, 0.0],
            )
            .unwrap();
        }
        #[test]
        fn dxdt_eq_2xt() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = Ode1be {
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |t: &TensorRank0, x: &TensorRank0| 2.0 * x * t,
                |t: &TensorRank0, _: &TensorRank0| 2.0 * t,
                0.0,
                1.0,
                &[0.0, 1.0],
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.powi(2).exp() - y).abs() < TOLERANCE
                        || (t.powi(2).exp() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn dxdt_eq_ix() -> Result<(), TestError> {
            let a = TensorRank2::<3, 1, 1>::identity();
            let (time, solution): (Vector, TensorRank1Vec<3, 1>) = Ode1be {
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, x: &TensorRank1<3, 1>| &a * x,
                |_: &TensorRank0, _: &TensorRank1<3, 1>| &a * 1.0,
                0.0,
                TensorRank1::new([1.0, 1.0, 1.0]),
                &[0.0, 1.0],
            )?;
            (0..3).for_each(|i| {
                time.iter().zip(solution.iter()).for_each(|(t, y)| {
                    assert!(
                        (t.exp() - y[i]).abs() < TOLERANCE
                            || (t.exp() / y[i] - 1.0).abs() < TOLERANCE
                    )
                })
            });
            Ok(())
        }
        #[test]
        fn first_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = Ode1be {
                opt_alg: $optimization,
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
                opt_alg: $optimization,
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
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<2, 1>| TensorRank1::new([y[1], -y[0]]),
                |_: &TensorRank0, _: &TensorRank1<2, 1>| {
                    TensorRank2::new([[0.0, 1.0], [-1.0, 0.0]])
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
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<3, 1>| TensorRank1::new([y[1], y[2], -y[1]]),
                |_: &TensorRank0, _: &TensorRank1<3, 1>| {
                    TensorRank2::new([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
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
                opt_alg: $optimization,
                ..Default::default()
            }
            .integrate(
                |_: &TensorRank0, y: &TensorRank1<4, 1>| TensorRank1::new([y[1], y[2], y[3], y[0]]),
                |_: &TensorRank0, _: &TensorRank1<4, 1>| {
                    TensorRank2::new([
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
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
