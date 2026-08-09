use super::LineSearch;
use crate::math::Scalar;

fn search(strong: bool) -> Scalar {
    LineSearch::Wolfe {
        control_1: 1e-3,
        control_2: 4e-1,
        cut_back: 5e-1,
        max_steps: 100,
        strong,
    }
    .backtrack(
        |x: &Scalar, _: Scalar| Ok(x.powi(2) / 2.0),
        |x: &Scalar| Ok(*x),
        &1.0,
        &1.0,
        &1.0,
        3.0,
    )
    .unwrap()
}

#[test]
fn wolfe_curvature_condition_follows_the_flag() {
    assert_eq!(search(false), 1.5);
    assert_eq!(search(true), 0.75)
}
