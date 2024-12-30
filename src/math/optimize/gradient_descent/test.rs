use super::{FirstOrder, GradientDescent, TensorRank0};

const TOLERANCE: TensorRank0 = 1e-5;

#[test]
fn linear() {
    assert!(
        GradientDescent {
            ..Default::default()
        }
        .minimize(|x: &TensorRank0| Ok(*x), 1.0, None, None)
        .unwrap()
        .abs()
            < TOLERANCE
    )
}

#[test]
fn quadratic() {
    assert!(
        GradientDescent {
            ..Default::default()
        }
        .minimize(|x: &TensorRank0| Ok(x.powi(2) / 2.0), 1.0, None, None)
        .unwrap()
        .abs()
            < TOLERANCE
    )
}

#[test]
fn sin() {
    assert!(
        GradientDescent {
            ..Default::default()
        }
        .minimize(|x: &TensorRank0| Ok(x.sin()), 1.0, None, None)
        .unwrap()
        .abs()
            < TOLERANCE
    )
}
