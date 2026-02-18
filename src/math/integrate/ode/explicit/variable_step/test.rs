macro_rules! test_explicit_variable_step {
    ($integration: expr) => {
        use crate::math::{
            Tensor, TensorArray, TensorRank1, TensorRank1Vec, TensorRank2, TensorTuple,
            TensorTupleVec, assert_eq_within_tols,
            integrate::{
                ode::explicit::test::test_explicit,
                test::{LENGTH, zero_to_one},
            },
        };
        test_explicit!($integration);
        #[test]
        fn dxdt_eq_neg_x() -> Result<(), TestError> {
            let (time, solution, function): (Vector, Vector, _) =
                $integration.integrate(|_: Scalar, x: &Scalar| Ok(-x), &[0.0, 0.8], 1.0)?;
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
                |t: Scalar, x: &Scalar| Ok(2.0 * x * t),
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
            let (time, solution, function): (Vector, Vector, _) =
                $integration.integrate(|t: Scalar, _: &Scalar| Ok(t.cos()), &[0.0, 1.0], 0.0)?;
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
                    |_: Scalar, x: &TensorRank1<3, 1>| Ok(&a * x),
                    &[0.0, 1.0],
                    TensorRank1::from([1.0, 1.0, 1.0]),
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
                |t: Scalar, _: &Scalar| Ok(t.cos()),
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
                    |t: Scalar, y: &TensorRank1<2, 1>| Ok(TensorRank1::from([y[1], -t.sin()])),
                    &[0.0, 6.0],
                    TensorRank1::from([0.0, 1.0]),
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
                    |t: Scalar, y: &TensorRank1<3, 1>| {
                        Ok(TensorRank1::from([y[1], y[2], -t.cos()]))
                    },
                    &[0.0, 1.0],
                    TensorRank1::from([0.0, 1.0, 0.0]),
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
                    |t: Scalar, y: &TensorRank1<4, 1>| {
                        Ok(TensorRank1::from([y[1], y[2], y[3], t.sin()]))
                    },
                    &[0.0, 0.6],
                    TensorRank1::from([0.0, 1.0, 0.0, -1.0]),
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
        #[test]
        fn flat() -> Result<(), TestError> {
            let (time, solution, function): (Vector, TensorRank1Vec<5, 1>, _) = $integration
                .integrate(
                    |t: Scalar, y: &TensorRank1<5, 1>| {
                        Ok(TensorRank1::from([y[1], -t.sin(), y[3], y[4], -t.cos()]))
                    },
                    &[0.0, 1.0],
                    TensorRank1::from([0.0, 1.0, 0.0, 1.0, 0.0]),
                )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    assert_eq_within_tols(&y[0], &t.sin())?;
                    assert_eq_within_tols(&f[0], &t.cos())?;
                    assert_eq_within_tols(&y[1], &t.cos())?;
                    assert_eq_within_tols(&f[1], &-t.sin())?;
                    assert_eq_within_tols(&y[2], &t.sin())?;
                    assert_eq_within_tols(&f[2], &t.cos())?;
                    assert_eq_within_tols(&y[3], &t.cos())?;
                    assert_eq_within_tols(&f[3], &-t.sin())?;
                    assert_eq_within_tols(&y[4], &-t.sin())?;
                    assert_eq_within_tols(&f[4], &-t.cos())
                })
        }
        #[test]
        fn tuple() -> Result<(), TestError> {
            let (time, solution, function): (
                Vector,
                TensorTupleVec<TensorRank1<2, 1>, TensorRank1<3, 1>>,
                _,
            ) = $integration.integrate(
                |t: Scalar, y: &TensorTuple<TensorRank1<2, 1>, TensorRank1<3, 1>>| {
                    let (y_1, y_2) = y.into();
                    Ok(TensorTuple::from((
                        TensorRank1::from([y_1[1], -t.sin()]),
                        TensorRank1::from([y_2[1], y_2[2], -t.cos()]),
                    )))
                },
                &[0.0, 1.0],
                TensorTuple::from((
                    TensorRank1::from([0.0, 1.0]),
                    TensorRank1::from([0.0, 1.0, 0.0]),
                )),
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    let (y_1, y_2) = y.into();
                    let (f_1, f_2) = f.into();
                    assert_eq_within_tols(&y_1[0], &t.sin())?;
                    assert_eq_within_tols(&f_1[0], &t.cos())?;
                    assert_eq_within_tols(&y_1[1], &t.cos())?;
                    assert_eq_within_tols(&f_1[1], &-t.sin())?;
                    assert_eq_within_tols(&y_2[0], &t.sin())?;
                    assert_eq_within_tols(&f_2[0], &t.cos())?;
                    assert_eq_within_tols(&y_2[1], &t.cos())?;
                    assert_eq_within_tols(&f_2[1], &-t.sin())?;
                    assert_eq_within_tols(&y_2[2], &-t.sin())?;
                    assert_eq_within_tols(&f_2[2], &-t.cos())
                })
        }
        #[test]
        fn tuple_nested() -> Result<(), TestError> {
            let (time, solution, function): (
                Vector,
                TensorTupleVec<
                    TensorRank1<2, 1>,
                    TensorTuple<TensorRank1<3, 1>, TensorRank1<4, 1>>,
                >,
                _,
            ) = $integration.integrate(
                |t: Scalar,
                 y: &TensorTuple<
                    TensorRank1<2, 1>,
                    TensorTuple<TensorRank1<3, 1>, TensorRank1<4, 1>>,
                >| {
                    let (y_1, y_23) = y.into();
                    let (y_2, y_3) = y_23.into();
                    Ok(TensorTuple::from((
                        TensorRank1::from([y_1[1], -t.sin()]),
                        TensorTuple::from((
                            TensorRank1::from([y_2[1], y_2[2], -t.cos()]),
                            TensorRank1::from([y_3[1], y_3[2], y_3[3], t.sin()]),
                        )),
                    )))
                },
                &[0.0, 0.6],
                TensorTuple::from((
                    TensorRank1::from([0.0, 1.0]),
                    TensorTuple::from((
                        TensorRank1::from([0.0, 1.0, 0.0]),
                        TensorRank1::from([0.0, 1.0, 0.0, -1.0]),
                    )),
                )),
            )?;
            time.iter()
                .zip(solution.iter().zip(function.iter()))
                .try_for_each(|(t, (y, f))| {
                    let (y_1, y_23) = y.into();
                    let (y_2, y_3) = y_23.into();
                    let (f_1, f_23) = f.into();
                    let (f_2, f_3) = f_23.into();
                    assert_eq_within_tols(&y_1[0], &t.sin())?;
                    assert_eq_within_tols(&f_1[0], &t.cos())?;
                    assert_eq_within_tols(&y_1[1], &t.cos())?;
                    assert_eq_within_tols(&f_1[1], &-t.sin())?;
                    assert_eq_within_tols(&y_2[0], &t.sin())?;
                    assert_eq_within_tols(&f_2[0], &t.cos())?;
                    assert_eq_within_tols(&y_2[1], &t.cos())?;
                    assert_eq_within_tols(&f_2[1], &-t.sin())?;
                    assert_eq_within_tols(&y_2[2], &-t.sin())?;
                    assert_eq_within_tols(&f_2[2], &-t.cos())?;
                    assert_eq_within_tols(&y_3[0], &t.sin())?;
                    assert_eq_within_tols(&f_3[0], &t.cos())?;
                    assert_eq_within_tols(&y_3[1], &t.cos())?;
                    assert_eq_within_tols(&f_3[1], &-t.sin())?;
                    assert_eq_within_tols(&y_3[2], &-t.sin())?;
                    assert_eq_within_tols(&f_3[2], &-t.cos())?;
                    assert_eq_within_tols(&y_3[3], &-t.cos())?;
                    assert_eq_within_tols(&f_3[3], &t.sin())
                })
        }
    };
}
pub(crate) use test_explicit_variable_step;
