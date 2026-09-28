use super::Vector;
use crate::math::Tensor;

fn vectors() -> (Vector, Vector) {
    (
        [1.0, -2.0, 3.5].into_iter().collect(),
        [0.5, 4.0, -1.0].into_iter().collect(),
    )
}

#[test]
fn a_borrowed_sum_matches_the_owned_sum() {
    let (a, b) = vectors();
    let expected = a.clone() + b.clone();
    assert_eq!(&a + &b, expected);
    assert_eq!(&a + b.clone(), expected);
    assert_eq!(
        expected.iter().copied().collect::<Vec<_>>(),
        [1.5, 2.0, 2.5]
    );
}
