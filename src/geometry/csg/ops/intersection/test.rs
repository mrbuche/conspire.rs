use super::Intersection;
use crate::{
    geometry::{csg::ops::test::check, mesh::test::cuboid, solid::SolidOracle},
    math::assert::AssertionError,
};

#[test]
fn intersection_matches_the_larger_distance() -> Result<(), AssertionError> {
    let (first, second) = (cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let oracle = Intersection::new(cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let (found, _) = check(&oracle, |point| {
        first
            .signed_distance(point)
            .max(second.signed_distance(point))
    })?;
    assert!(found > 0);
    Ok(())
}

#[test]
fn nested_intersection_is_the_inner_solid() -> Result<(), AssertionError> {
    let inner = cuboid([0.3; 3], [0.7; 3]);
    let oracle = Intersection::new(cuboid([0.0; 3], [1.0; 3]), cuboid([0.3; 3], [0.7; 3]));
    let (found, missing) = check(&oracle, |point| inner.signed_distance(point))?;
    assert!(found > 0);
    assert_eq!(missing, 0);
    Ok(())
}
