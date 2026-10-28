use super::Union;
use crate::{
    geometry::{csg::ops::test::check, mesh::test::cuboid, solid::SolidOracle},
    math::assert::AssertionError,
};

#[test]
fn overlapping_union_matches_the_smaller_distance() -> Result<(), AssertionError> {
    let (first, second) = (cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let oracle = Union::new(cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let (found, _) = check(&oracle, |point| {
        first
            .signed_distance(point)
            .min(second.signed_distance(point))
    })?;
    assert!(found > 0);
    Ok(())
}

#[test]
fn disjoint_union_always_projects() -> Result<(), AssertionError> {
    let (first, second) = (cuboid([0.0; 3], [0.2; 3]), cuboid([0.6; 3], [1.0; 3]));
    let oracle = Union::new(cuboid([0.0; 3], [0.2; 3]), cuboid([0.6; 3], [1.0; 3]));
    let (found, missing) = check(&oracle, |point| {
        first
            .signed_distance(point)
            .min(second.signed_distance(point))
    })?;
    assert!(found > 0);
    assert_eq!(missing, 0);
    Ok(())
}
