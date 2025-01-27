use super::{
    super::{
        super::{
            test::TestError, Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1Vec, Vector,
        },
        test::zero_to_tau,
    },
    Explicit, Ode23,
};
use std::f64::consts::TAU;

const LENGTH: usize = 33;
const TOLERANCE: TensorRank0 = 5.0 * crate::ABS_TOL;

#[test]
#[should_panic(expected = "The time must contain at least two entries.")]
fn initial_time_not_less_than_final_time() {
    let _: (Vector, Vector) = Ode23 {
        ..Default::default()
    }
    .integrate(|t: &TensorRank0, _: &TensorRank0| t.cos(), 0.0, 0.0, &[0.0])
    .unwrap();
}

#[test]
#[should_panic(expected = "The initial time must precede the final time.")]
fn length_time_less_than_two() {
    let _: (Vector, Vector) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, _: &TensorRank0| t.cos(),
        0.0,
        0.0,
        &[0.0, 1.0, 0.0],
    )
    .unwrap();
}

#[test]
fn dxdt_eq_2xt() -> Result<(), TestError> {
    let (time, solution): (Vector, Vector) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, x: &TensorRank0| 2.0 * x * t,
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
fn first_order_tensor_rank_0() -> Result<(), TestError> {
    let (time, solution): (Vector, Vector) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, _: &TensorRank0| t.cos(),
        0.0,
        0.0,
        &[0.0, TAU],
    )?;
    time.iter().zip(solution.iter()).for_each(|(t, y)| {
        assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
    });
    Ok(())
}

#[test]
fn first_order_tensor_rank_0_eval_times() -> Result<(), TestError> {
    let (time, solution): (Vector, Vector) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, _: &TensorRank0| t.cos(),
        0.0,
        0.0,
        &zero_to_tau::<LENGTH>(),
    )?;
    time.iter().zip(solution.iter()).for_each(|(t, y)| {
        assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
    });
    Ok(())
}

#[test]
fn second_order_tensor_rank_0() -> Result<(), TestError> {
    let (time, solution): (Vector, TensorRank1Vec<2, 1>) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, y: &TensorRank1<2, 1>| TensorRank1::new([y[1], -t.sin()]),
        0.0,
        TensorRank1::new([0.0, 1.0]),
        &[0.0, TAU],
    )?;
    time.iter().zip(solution.iter()).for_each(|(t, y)| {
        assert!((t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE)
    });
    Ok(())
}

#[test]
fn third_order_tensor_rank_0() -> Result<(), TestError> {
    let (time, solution): (Vector, TensorRank1Vec<3, 1>) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, y: &TensorRank1<3, 1>| TensorRank1::new([y[1], y[2], -t.cos()]),
        0.0,
        TensorRank1::new([0.0, 1.0, 0.0]),
        &[0.0, TAU],
    )?;
    time.iter().zip(solution.iter()).for_each(|(t, y)| {
        assert!((t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE)
    });
    Ok(())
}

#[test]
fn fourth_order_tensor_rank_0() -> Result<(), TestError> {
    let (time, solution): (Vector, TensorRank1Vec<4, 1>) = Ode23 {
        ..Default::default()
    }
    .integrate(
        |t: &TensorRank0, y: &TensorRank1<4, 1>| TensorRank1::new([y[1], y[2], y[3], t.sin()]),
        0.0,
        TensorRank1::new([0.0, 1.0, 0.0, -1.0]),
        &[0.0, TAU],
    )?;
    time.iter().zip(solution.iter()).for_each(|(t, y)| {
        assert!((t.sin() - y[0]).abs() < TOLERANCE || (t.sin() / y[0] - 1.0).abs() < TOLERANCE)
    });
    Ok(())
}
