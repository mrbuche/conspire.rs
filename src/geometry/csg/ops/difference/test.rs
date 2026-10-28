use super::Difference;
use crate::{
    geometry::{
        Coordinate, Direction, csg::ops::test::check, mesh::test::cuboid, solid::SolidOracle,
    },
    math::assert::{Assert, AssertionError},
};

#[test]
fn overlapping_difference_matches_outer_minus_inner() -> Result<(), AssertionError> {
    let (outer, inner) = (cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let oracle = Difference::new(cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3]));
    let (found, _) = check(&oracle, |point| {
        outer
            .signed_distance(point)
            .max(-inner.signed_distance(point))
    })?;
    assert!(found > 0);
    Ok(())
}

#[test]
fn cavity_wall_normal_points_into_the_cavity() -> Result<(), AssertionError> {
    let oracle = Difference::new(cuboid([0.0; 3], [1.0; 3]), cuboid([0.3; 3], [0.7; 3]));
    let (_, cavity) = oracle.project(&Coordinate::from([0.5, 0.5, 0.25])).unwrap();
    Assert::default().eq_within_tols(&cavity, &Direction::from([0.0, 0.0, 1.0]))?;
    let (_, outer) = oracle.project(&Coordinate::from([0.5, 0.5, 0.05])).unwrap();
    Assert::default().eq_within_tols(&outer, &Direction::from([0.0, 0.0, -1.0]))
}

#[test]
fn nested_difference_always_projects() -> Result<(), AssertionError> {
    let (outer, inner) = (cuboid([0.0; 3], [1.0; 3]), cuboid([0.3; 3], [0.7; 3]));
    let oracle = Difference::new(cuboid([0.0; 3], [1.0; 3]), cuboid([0.3; 3], [0.7; 3]));
    let (_, missing) = check(&oracle, |point| {
        outer
            .signed_distance(point)
            .max(-inner.signed_distance(point))
    })?;
    assert_eq!(missing, 0);
    Ok(())
}
