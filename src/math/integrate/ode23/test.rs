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

const TOLERANCE: TensorRank0 = 1e-5;

// #[test]
// #[should_panic(expected = "Evaluation times precede the initial time.")]
// fn evaluation_times_precede_initial_time() {
//     let mut evaluation_times = zero_to_tau::<LENGTH>();
//     evaluation_times[0] = -1.0;
//     let _: TensorRank0List<LENGTH> = Ode23 {
//         ..Default::default()
//     }
//     .integrate(
//         |t: &TensorRank0, _: &TensorRank0| t.cos(),
//         0.0,
//         0.0,
//         &evaluation_times,
//     )
//     .unwrap();
// }

// #[test]
// #[should_panic(expected = "Evaluation times must include a final time.")]
// fn evaluation_times_no_final_time() {
//     let _: TensorRank0List<LENGTH> = Ode23 {
//         ..Default::default()
//     }
//     .integrate(
//         |t: &TensorRank0, _: &TensorRank0| t.cos(),
//         0.0,
//         0.0,
//         &TensorRank0List::new([0.0]),
//     )
//     .unwrap();
// }

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

// #[test]
// fn first_order_tensor_rank_0_one_evaluation_time_after_initial_time() -> Result<(), TestError> {
//     let evaluation_times = TensorRank0List::new([1.0]);
//     let solution: TensorRank0List<LENGTH> = Ode23 {
//         ..Default::default()
//     }
//     .integrate(
//         |t: &TensorRank0, _: &TensorRank0| t.cos(),
//         0.0,
//         0.0,
//         &evaluation_times,
//     )?;
//     evaluation_times
//         .iter()
//         .zip(solution.iter())
//         .for_each(|(t, y)| {
//             assert!((t.sin() - y).abs() < TOLERANCE || (t.sin() / y - 1.0).abs() < TOLERANCE)
//         });
//     Ok(())
// }

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
