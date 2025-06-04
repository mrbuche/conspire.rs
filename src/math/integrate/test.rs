use super::IntegrationError;
use crate::math::{TensorArray, TensorRank0, TensorRank0List};

pub const LENGTH: usize = 33;

pub fn zero_to_one<const W: usize>() -> [TensorRank0; W] {
    (0..W)
        .map(|i| (i as TensorRank0) / ((W - 1) as TensorRank0))
        .collect::<TensorRank0List<W>>()
        .as_array()
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
                Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1Vec, TensorRank2, Vector,
                assert_eq_within_tols, test::TestError,
            },
            Explicit, IntegrationError,
            test::{LENGTH, zero_to_one},
        };
        #[test]
        #[should_panic(expected = "The time must contain at least two entries.")]
        fn initial_time_not_less_than_final_time() {
            let _: (Vector, Vector, _) = $integration
                .integrate(|_: TensorRank0, _: &TensorRank0| panic!(), &[0.0], 0.0)
                .unwrap();
        }
        #[test]
        fn into_test_error() {
            let result: Result<(Vector, Vector, _), IntegrationError> =
                $integration.integrate(|_: TensorRank0, _: &TensorRank0| panic!(), &[0.0], 0.0);
            let _: TestError = result.unwrap_err().into();
        }
        #[test]
        #[should_panic(expected = "The initial time must precede the final time.")]
        fn length_time_less_than_two() {
            let _: (Vector, Vector, _) = $integration
                .integrate(
                    |_: TensorRank0, _: &TensorRank0| panic!(),
                    &[0.0, 1.0, 0.0],
                    0.0,
                )
                .unwrap();
        }
        #[test]
        fn dxdt_eq_neg_x() -> Result<(), TestError> {
            let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                |_: TensorRank0, x: &TensorRank0| Ok(-x),
                &[0.0, 0.8],
                1.0,
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(y, &(-t).exp())?;
                    assert_eq_within_tols(f, &-y)
                })
        }
        #[test]
        fn dxdt_eq_2xt() -> Result<(), TestError> {
            let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                |t: TensorRank0, x: &TensorRank0| Ok(2.0 * x * t),
                &[0.0, 1.0],
                1.0,
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(y, &t.powi(2).exp())?;
                    assert_eq_within_tols(f, &(2.0 * y * t))
                })
        }
        #[test]
        fn dxdt_eq_cos_t() -> Result<(), TestError> {
            let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                |t: TensorRank0, _: &TensorRank0| Ok(t.cos()),
                &[0.0, 1.0],
                0.0,
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(y, &t.sin())?;
                    assert_eq_within_tols(f, &t.cos())
                })
        }
        #[test]
        fn dxdt_eq_ix() -> Result<(), TestError> {
            let a = TensorRank2::<3, 1, 1>::identity();
            let (time, solution, function): (Vector, TensorRank1Vec<3, 1>, _) = $integration
                .integrate(
                    |_: TensorRank0, x: &TensorRank1<3, 1>| Ok(&a * x),
                    &[0.0, 1.0],
                    TensorRank1::new([1.0, 1.0, 1.0]),
                )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    y.iter().zip(f.iter()).try_for_each(|(y_n, f_n)| {
                        assert_eq_within_tols(y_n, &t.exp())?;
                        assert_eq_within_tols(f_n, y_n)
                    })
                })
        }
        #[test]
        fn eval_times() -> Result<(), TestError> {
            let (time, solution, function): (Vector, Vector, _) = $integration.integrate(
                |t: TensorRank0, _: &TensorRank0| Ok(t.cos()),
                &zero_to_one::<LENGTH>(),
                0.0,
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(y, &t.sin())?;
                    assert_eq_within_tols(f, &t.cos())
                })
        }
        #[test]
        fn second_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution, function): (Vector, TensorRank1Vec<2, 1>, _) = $integration
                .integrate(
                    |t: TensorRank0, y: &TensorRank1<2, 1>| Ok(TensorRank1::new([y[1], -t.sin()])),
                    &[0.0, 6.0],
                    TensorRank1::new([0.0, 1.0]),
                )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(&y[0], &t.sin())?;
                    assert_eq_within_tols(&f[0], &t.cos())?;
                    assert_eq_within_tols(&y[1], &t.cos())?;
                    assert_eq_within_tols(&f[1], &-t.sin())
                })
        }
        #[test]
        fn third_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution, function): (Vector, TensorRank1Vec<3, 1>, _) = $integration
                .integrate(
                    |t: TensorRank0, y: &TensorRank1<3, 1>| {
                        Ok(TensorRank1::new([y[1], y[2], -t.cos()]))
                    },
                    &[0.0, 1.0],
                    TensorRank1::new([0.0, 1.0, 0.0]),
                )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(&y[0], &t.sin())?;
                    assert_eq_within_tols(&f[0], &t.cos())?;
                    assert_eq_within_tols(&y[1], &t.cos())?;
                    assert_eq_within_tols(&f[1], &-t.sin())?;
                    assert_eq_within_tols(&y[2], &-t.sin())?;
                    assert_eq_within_tols(&f[2], &-t.cos())
                })
        }
        #[test]
        fn fourth_order_tensor_rank_0() -> Result<(), TestError> {
            let (time, solution, function): (Vector, TensorRank1Vec<4, 1>, _) = $integration
                .integrate(
                    |t: TensorRank0, y: &TensorRank1<4, 1>| {
                        Ok(TensorRank1::new([y[1], y[2], y[3], t.sin()]))
                    },
                    &[0.0, 0.6],
                    TensorRank1::new([0.0, 1.0, 0.0, -1.0]),
                )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(&y[0], &t.sin())?;
                    assert_eq_within_tols(&f[0], &t.cos())?;
                    assert_eq_within_tols(&y[1], &t.cos())?;
                    assert_eq_within_tols(&f[1], &-t.sin())?;
                    assert_eq_within_tols(&y[2], &-t.sin())?;
                    assert_eq_within_tols(&f[2], &-t.cos())?;
                    assert_eq_within_tols(&y[3], &-t.cos())?;
                    assert_eq_within_tols(&f[3], &t.sin())
                })
        }
    };
}
pub(crate) use test_explicit;
