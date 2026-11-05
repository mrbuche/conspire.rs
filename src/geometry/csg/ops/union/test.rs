use super::Union;
use crate::{
    geometry::{csg::ops::test::check, mesh::test::solid, solid::SolidOracle},
    math::assert::AssertionError,
};

#[test]
fn overlapping_union_matches_the_smaller_distance() -> Result<(), AssertionError> {
    let (first, second) = (solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let oracle = Union::new(solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let counts = check(&oracle, |point| {
        first
            .signed_distance(point)
            .min(second.signed_distance(point))
    })?;
    assert!(counts.found() > 0);
    Ok(())
}

#[test]
fn disjoint_union_always_projects() -> Result<(), AssertionError> {
    let (first, second) = (solid([0.0; 3], [0.2; 3]), solid([0.6; 3], [1.0; 3]));
    let oracle = Union::new(solid([0.0; 3], [0.2; 3]), solid([0.6; 3], [1.0; 3]));
    let counts = check(&oracle, |point| {
        first
            .signed_distance(point)
            .min(second.signed_distance(point))
    })?;
    assert!(counts.exact > 0);
    assert_eq!(counts.inexact, 0);
    assert_eq!(counts.missing, 0);
    Ok(())
}
