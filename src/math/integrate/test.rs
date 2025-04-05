use super::IntegrationError;
use crate::math::{test::TestError, TensorArray, TensorRank0, TensorRank0List};
use std::f64::consts::TAU;

pub const LENGTH: usize = 33;

pub fn zero_to_tau<const W: usize>() -> [TensorRank0; W] {
    (0..W)
        .map(|i| TAU * (i as TensorRank0) / ((W - 1) as TensorRank0))
        .collect::<TensorRank0List<W>>()
        .as_array()
}

impl From<IntegrationError> for TestError {
    fn from(error: IntegrationError) -> TestError {
        TestError {
            message: format!("{}", error),
        }
    }
}

#[test]
fn debug() {
    let _ = format!("{:?}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{:?}", IntegrationError::LengthTimeLessThanTwo);
}

#[test]
fn display() {
    let _ = format!("{}", IntegrationError::InitialTimeNotLessThanFinalTime);
    let _ = format!("{}", IntegrationError::LengthTimeLessThanTwo);
}

macro_rules! test_explicit {
    ($integration: expr) => {
        use super::super::{
            super::{
                test::TestError, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1Vec,
                TensorRank2, Vector,
            },
            test::{zero_to_tau, LENGTH},
            Explicit, IntegrationError
        };
        use std::f64::consts::TAU;
        pub const TOLERANCE: TensorRank0 = 5.0 * crate::ABS_TOL;
        #[test]
        #[should_panic(expected = "The time must contain at least two entries.")]
        fn initial_time_not_less_than_final_time() {
            let _: (Vector, Vector) = $integration
                .integrate(|_: &TensorRank0, _: &TensorRank0| panic!(), &[0.0], 0.0)
                .unwrap();
        }
        #[test]
        fn into_test_error() {
            let result: Result<(Vector, Vector), IntegrationError> = $integration.integrate(|_: &TensorRank0, _: &TensorRank0| panic!(), &[0.0], 0.0);
            let _: TestError = result.unwrap_err().into();
        }
        #[test]
        #[should_panic(expected = "The initial time must precede the final time.")]
        fn length_time_less_than_two() {
            let _: (Vector, Vector) = $integration
                .integrate(
                    |_: &TensorRank0, _: &TensorRank0| panic!(),
                    &[0.0, 1.0, 0.0],
                    0.0,
                )
                .unwrap();
        }
        #[test]
        fn dxdt_eq_neg_x() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) =
                $integration.integrate(|_: &TensorRank0, x: &TensorRank0| -x, &[0.0, 1.0], 1.0)?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    ((-t).exp() - y).abs() < TOLERANCE // || ((-t).exp() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn dxdt_eq_2xt() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = $integration.integrate(
                |t: &TensorRank0, x: &TensorRank0| 2.0 * x * t,
                &[0.0, 1.0],
                1.0,
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.powi(2).exp() - y).abs() < TOLERANCE // || (t.powi(2).exp() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn dxdt_eq_cos_t() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = $integration.integrate(
                |t: &TensorRank0, _: &TensorRank0| t.cos(),
                &[0.0, 16.0 * TAU],
                0.0,
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y).abs() < TOLERANCE // || (t.sin() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn dxdt_eq_ix() -> Result<(), TestError> {
            let a = TensorRank2::<3, 1, 1>::identity();
            let (time, solution): (Vector, TensorRank1Vec<3, 1>) = $integration.integrate(
                |_: &TensorRank0, x: &TensorRank1<3, 1>| &a * x,
                &[0.0, 1.0],
                TensorRank1::new([1.0, 1.0, 1.0]),
            )?;
            (0..3).for_each(|i| {
                time.iter().zip(solution.iter()).for_each(|(t, y)| {
                    assert!(
                        (t.exp() - y[i]).abs() < TOLERANCE // || (t.exp() / y[i] - 1.0).abs() < TOLERANCE
                    )
                })
            });
            Ok(())
        }
        #[test]
        fn first_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = $integration.integrate(
                |t: &TensorRank0, _: &TensorRank0| t.cos(),
                &[0.0, TAU],
                0.0,
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y).abs() < TOLERANCE // || (t.sin() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn first_order_tensor_rank_0_eval_times() -> Result<(), TestError> {
            let (time, solution): (Vector, Vector) = $integration.integrate(
                |t: &TensorRank0, _: &TensorRank0| t.cos(),
                &zero_to_tau::<LENGTH>(),
                0.0,
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y).abs() < TOLERANCE // || (t.sin() / y - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn second_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<2, 1>) = $integration.integrate(
                |t: &TensorRank0, y: &TensorRank1<2, 1>| TensorRank1::new([y[1], -t.sin()]),
                &[0.0, TAU],
                TensorRank1::new([0.0, 1.0]),
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE // || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn third_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<3, 1>) = $integration.integrate(
                |t: &TensorRank0, y: &TensorRank1<3, 1>| TensorRank1::new([y[1], y[2], -t.cos()]),
                &[0.0, TAU],
                TensorRank1::new([0.0, 1.0, 0.0]),
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE // || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
        #[test]
        fn fourth_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution): (Vector, TensorRank1Vec<4, 1>) = $integration.integrate(
                |t: &TensorRank0, y: &TensorRank1<4, 1>| {
                    TensorRank1::new([y[1], y[2], y[3], t.sin()])
                },
                &[0.0, TAU],
                TensorRank1::new([0.0, 1.0, 0.0, -1.0]),
            )?;
            time.iter().zip(solution.iter()).for_each(|(t, y)| {
                assert!(
                    (t.sin() - y[0]).abs() < TOLERANCE // || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
                )
            });
            Ok(())
        }
    };
}
pub(crate) use test_explicit;

// macro_rules! test_implicit {
//     ($integration: expr) => {
//         use super::super::super::{
//             super::{
//                 test::TestError, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1Vec,
//                 TensorRank2, Vector,
//             },
//             test::{zero_to_tau, LENGTH},
//             Implicit,
//         };
//         use std::f64::consts::TAU;
//         // pub const TOLERANCE: TensorRank0 = 1e-5;
//         #[test]
//         #[should_panic(expected = "The time must contain at least two entries.")]
//         fn initial_time_not_less_than_final_time() {
//             let _: (Vector, Vector) = $integration
//                 .integrate(
//                     |t: &TensorRank0, _: &TensorRank0| t.cos(),
//                     |_: &TensorRank0, _: &TensorRank0| 0.0,
//                     &[0.0],
//                     0.0,
//                 )
//                 .unwrap();
//         }
//         #[test]
//         #[should_panic(expected = "The initial time must precede the final time.")]
//         fn length_time_less_than_two() {
//             let _: (Vector, Vector) = $integration
//                 .integrate(
//                     |t: &TensorRank0, _: &TensorRank0| t.cos(),
//                     |_: &TensorRank0, _: &TensorRank0| 0.0,
//                     &[0.0, 1.0, 0.0],
//                     0.0,
//                 )
//                 .unwrap();
//         }
//         #[test]
//         fn dxdt_eq_neg_x() -> Result<(), TestError> {
//             let (time, solution): (Vector, Vector) = $integration.integrate(
//                 |_: &TensorRank0, x: &TensorRank0| -x,
//                 |_: &TensorRank0, _: &TensorRank0| -1.0,
//                 &[0.0, 1.0],
//                 1.0,
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!(
//                     ((-t).exp() - y).abs() < TOLERANCE || ((-t).exp() / y - 1.0).abs() < TOLERANCE
//                 )
//             });
//             Ok(())
//         }
//         #[test]
//         fn dxdt_eq_2xt() -> Result<(), TestError> {
//             let (time, solution): (Vector, Vector) = $integration.integrate(
//                 |t: &TensorRank0, x: &TensorRank0| 2.0 * x * t,
//                 |t: &TensorRank0, _: &TensorRank0| 2.0 * t,
//                 &[0.0, 1.0],
//                 1.0,
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!(
//                     (t.powi(2).exp() - y).abs() < TOLERANCE
//                         || (t.powi(2).exp() / y - 1.0).abs() < TOLERANCE
//                 )
//             });
//             Ok(())
//         }
//         #[test]
//         fn dxdt_eq_cos_t() -> Result<(), TestError> {
//             let (time, solution): (Vector, Vector) = $integration.integrate(
//                 |t: &TensorRank0, _: &TensorRank0| t.cos(),
//                 |_: &TensorRank0, _: &TensorRank0| 0.0,
//                 &[0.0, 16.0 * TAU],
//                 0.0,
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
//             });
//             Ok(())
//         }
//         #[test]
//         fn dxdt_eq_ix() -> Result<(), TestError> {
//             let a = TensorRank2::<3, 1, 1>::identity();
//             let (time, solution): (Vector, TensorRank1Vec<3, 1>) = $integration.integrate(
//                 |_: &TensorRank0, x: &TensorRank1<3, 1>| &a * x,
//                 |_: &TensorRank0, _: &TensorRank1<3, 1>| a.clone(),
//                 &[0.0, 1.0],
//                 TensorRank1::new([1.0, 1.0, 1.0]),
//             )?;
//             (0..3).for_each(|i| {
//                 time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                     assert!(
//                         (t.exp() - y[i]).abs() < TOLERANCE
//                             || (t.exp() / y[i] - 1.0).abs() < TOLERANCE
//                     )
//                 })
//             });
//             Ok(())
//         }
//         #[test]
//         fn first_order_tensor_rank_0() -> Result<(), TestError> {
//             let (time, solution): (Vector, Vector) = $integration.integrate(
//                 |t: &TensorRank0, _: &TensorRank0| t.cos(),
//                 |_: &TensorRank0, _: &TensorRank0| 0.0,
//                 &[0.0, TAU],
//                 0.0,
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
//             });
//             Ok(())
//         }
//         #[test]
//         fn first_order_tensor_rank_0_eval_times() -> Result<(), TestError> {
//             let (time, solution): (Vector, Vector) = $integration.integrate(
//                 |t: &TensorRank0, _: &TensorRank0| t.cos(),
//                 |_: &TensorRank0, _: &TensorRank0| 0.0,
//                 &zero_to_tau::<LENGTH>(),
//                 0.0,
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
//             });
//             Ok(())
//         }
//         #[test]
//         fn second_order_tensor_rank_0() -> Result<(), TestError> {
//             let (time, solution): (Vector, TensorRank1Vec<2, 1>) = $integration.integrate(
//                 |_: &TensorRank0, y: &TensorRank1<2, 1>| TensorRank1::new([y[1], -y[0]]),
//                 |_: &TensorRank0, _: &TensorRank1<2, 1>| {
//                     TensorRank2::new([[0.0, 1.0], [-1.0, 0.0]])
//                 },
//                 &[0.0, TAU],
//                 TensorRank1::new([0.0, 1.0]),
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!(
//                     (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
//                 )
//             });
//             Ok(())
//         }
//         #[test]
//         fn third_order_tensor_rank_0() -> Result<(), TestError> {
//             let (time, solution): (Vector, TensorRank1Vec<3, 1>) = $integration.integrate(
//                 |_: &TensorRank0, y: &TensorRank1<3, 1>| TensorRank1::new([y[1], y[2], -y[1]]),
//                 |_: &TensorRank0, _: &TensorRank1<3, 1>| {
//                     TensorRank2::new([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
//                 },
//                 &[0.0, TAU],
//                 TensorRank1::new([0.0, 1.0, 0.0]),
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!(
//                     (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
//                 )
//             });
//             Ok(())
//         }
//         #[test]
//         fn fourth_order_tensor_rank_0() -> Result<(), TestError> {
//             let (time, solution): (Vector, TensorRank1Vec<4, 1>) = $integration.integrate(
//                 |_: &TensorRank0, y: &TensorRank1<4, 1>| TensorRank1::new([y[1], y[2], y[3], y[0]]),
//                 |_: &TensorRank0, _: &TensorRank1<4, 1>| {
//                     TensorRank2::new([
//                         [0.0, 1.0, 0.0, 0.0],
//                         [0.0, 0.0, 1.0, 0.0],
//                         [0.0, 0.0, 0.0, 1.0],
//                         [1.0, 0.0, 0.0, 0.0],
//                     ])
//                 },
//                 &[0.0, TAU],
//                 TensorRank1::new([0.0, 1.0, 0.0, -1.0]),
//             )?;
//             time.iter().zip(solution.iter()).for_each(|(t, y)| {
//                 assert!(
//                     (t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE
//                 )
//             });
//             Ok(())
//         }
//     };
// }
// pub(crate) use test_implicit;
