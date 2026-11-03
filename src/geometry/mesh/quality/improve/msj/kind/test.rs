use super::KINDS;
use crate::math::{
    Scalar,
    assert::{Assert, AssertionError},
};

#[test]
fn each_kind_counts_its_points_and_edges() {
    assert_eq!(
        KINDS
            .iter()
            .map(|kind| (kind.points.len(), kind.edges.len()))
            .collect::<Vec<_>>(),
        [(9, 12), (4, 6), (6, 9), (8, 8)]
    )
}

#[test]
fn the_points_of_every_kind_are_differences_of_nodes() -> Result<(), AssertionError> {
    for kind in KINDS.iter() {
        for point in kind.points {
            for (row, terms) in point.weights.iter().zip(&point.sparse) {
                Assert::default().eq_within_tols(row.iter().sum::<Scalar>(), 0.0)?;
                assert_eq!(
                    terms.len(),
                    row.iter().filter(|&&weight| weight != 0.0).count()
                )
            }
        }
    }
    Ok(())
}
