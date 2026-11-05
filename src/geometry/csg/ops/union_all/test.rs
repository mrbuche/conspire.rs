use super::UnionAll;
use crate::{
    geometry::{
        Coordinate,
        csg::ops::{
            Union,
            test::{check, grid},
        },
        mesh::{Watertight, test::solid},
        solid::SolidOracle,
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

fn boxes() -> Vec<Watertight> {
    vec![
        solid([0.0; 3], [0.5; 3]),
        solid([0.3; 3], [0.8; 3]),
        solid([0.6; 3], [1.0; 3]),
    ]
}

#[test]
fn union_all_matches_the_smallest_distance() -> Result<(), AssertionError> {
    let solids = boxes();
    let oracle = UnionAll::new(boxes());
    let counts = check(&oracle, |point| {
        solids
            .iter()
            .map(|solid| solid.signed_distance(point))
            .fold(Length::meters(f64::INFINITY), |a, b| a.min(b))
    })?;
    assert!(counts.found() > 0);
    Ok(())
}

#[test]
fn union_all_of_two_is_the_binary_union() -> Result<(), AssertionError> {
    let all = UnionAll::new(vec![solid([0.0; 3], [0.5; 3]), solid([0.3; 3], [0.8; 3])]);
    let pair = Union::new(solid([0.0; 3], [0.5; 3]), solid([0.3; 3], [0.8; 3]));
    grid().iter().try_for_each(|point| {
        Assert::default().eq_within_tols(all.signed_distance(point), &pair.signed_distance(point))
    })
}

#[test]
fn the_empty_union_is_the_empty_set() {
    let oracle = UnionAll::<Watertight>::new(Vec::new());
    let point = Coordinate::from([0.5; 3]);
    assert!(oracle.signed_distance(&point).value().is_infinite());
    assert!(oracle.project(&point).is_none());
}
