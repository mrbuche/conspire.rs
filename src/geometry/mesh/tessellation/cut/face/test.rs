use super::{super::Sign, face_cut};
use crate::geometry::Coordinate;
use std::collections::HashMap;

#[test]
fn face_cut_pentagon() {
    let corners = [0, 1, 2, 3, 4];
    let signs: HashMap<usize, Sign> = [
        (0, Sign::Inside),
        (1, Sign::Inside),
        (2, Sign::Outside),
        (3, Sign::Outside),
        (4, Sign::Outside),
    ]
    .into_iter()
    .collect();
    let point = Coordinate::const_from([0.0, 0.0, 0.0]);
    let crossings = [([1, 2], vec![point.clone()]), ([0, 4], vec![point])]
        .into_iter()
        .collect();
    let cut = face_cut(&corners, &signs, &crossings).unwrap();
    assert_eq!(cut.endpoints.len(), 2);
    assert!(cut.inside);
    assert!(!cut.flush);
}
