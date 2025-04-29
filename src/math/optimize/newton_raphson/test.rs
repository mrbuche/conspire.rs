use super::{NewtonRaphson, FirstOrderRootFinding, TensorRank0};

const TOLERANCE: TensorRank0 = 1e-6;

#[test]
fn linear() {
    assert!(
        NewtonRaphson {
            ..Default::default()
        }
        .root(
            |x: &TensorRank0| Ok(*x),
            |_: &TensorRank0| Ok(1.0),
            1.0,
        )
        .unwrap()
        .abs()
            < TOLERANCE
    )
}

#[test]
fn quadratic() {
    assert!(
        NewtonRaphson {
            ..Default::default()
        }
        .root(
            |x: &TensorRank0| Ok(x.powi(2) / 2.0),
            |x: &TensorRank0| Ok(*x),
            1.0,
        )
        .unwrap()
        .abs()
            < TOLERANCE
    )
}

#[test]
fn sin() {
    assert!(
        NewtonRaphson {
            ..Default::default()
        }
        .root(
            |x: &TensorRank0| Ok(x.sin()),
            |x: &TensorRank0| Ok(x.cos()),
            1.0,
        )
        .unwrap()
        .abs()
            < TOLERANCE
    )
}

// #[test]
// #[should_panic(expected = "The obtained solution is not a minimum.")]
// fn sin_max() {
//     NewtonRaphson {
//         ..Default::default()
//     }
//     .root(
//         |x: &TensorRank0| Ok(x.sin()),
//         |x: &TensorRank0| Ok(x.cos()),
//         3.0,
//     )
//     .unwrap();
// }
