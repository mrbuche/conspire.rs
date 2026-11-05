use super::Intersection;
use crate::{
    geometry::{csg::ops::test::check, mesh::test::solid, solid::SolidOracle},
    math::assert::AssertionError,
};

#[test]
fn intersection_matches_the_larger_distance() -> Result<(), AssertionError> {
    let (first, second) = (solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let oracle = Intersection::new(solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let counts = check(&oracle, |point| {
        first
            .signed_distance(point)
            .max(second.signed_distance(point))
    })?;
    assert!(counts.found() > 0);
    Ok(())
}

#[test]
fn nested_intersection_is_the_inner_solid() -> Result<(), AssertionError> {
    let inner = solid([0.3; 3], [0.7; 3]);
    let oracle = Intersection::new(solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let counts = check(&oracle, |point| inner.signed_distance(point))?;
    assert!(counts.exact > 0);
    assert!(counts.inexact > 0);
    assert_eq!(counts.missing, 0);
    Ok(())
}
