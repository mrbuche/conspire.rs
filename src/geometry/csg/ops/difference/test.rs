use super::Difference;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::ops::{
            Union,
            test::{check, grid},
        },
        mesh::test::solid,
        solid::SolidOracle,
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
};

#[test]
fn overlapping_difference_matches_outer_minus_inner() -> Result<(), AssertionError> {
    let (outer, inner) = (solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let oracle = Difference::new(solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let counts = check(&oracle, |point| {
        outer
            .signed_distance(point)
            .max(-inner.signed_distance(point))
    })?;
    assert!(counts.found() > 0);
    Ok(())
}

#[test]
fn cavity_wall_normal_points_into_the_cavity() -> Result<(), AssertionError> {
    let oracle = Difference::new(solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let cavity = oracle
        .project(&Coordinate::from([0.5, 0.5, 0.25]))
        .unwrap()
        .normal;
    Assert::default().eq_within_tols(&cavity, &Direction::from([0.0, 0.0, 1.0]))?;
    let outer = oracle
        .project(&Coordinate::from([0.5, 0.5, 0.05]))
        .unwrap()
        .normal;
    Assert::default().eq_within_tols(&outer, &Direction::from([0.0, 0.0, -1.0]))
}

#[test]
fn nested_difference_always_projects() -> Result<(), AssertionError> {
    let (outer, inner) = (solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let oracle = Difference::new(solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let counts = check(&oracle, |point| {
        outer
            .signed_distance(point)
            .max(-inner.signed_distance(point))
    })?;
    assert_eq!(counts.missing, 0);
    assert_eq!(counts.inexact, 0);
    Ok(())
}

#[test]
fn an_inexact_operand_makes_the_difference_inexact() {
    let union = || Union::new(solid([0.0; 3], [0.7; 3]), solid([0.3; 3], [1.0; 3]));
    let difference = Difference::new(union(), solid([5.0; 3], [5.2; 3]));
    let points = grid();
    let inexact: Vec<_> = points
        .iter()
        .filter(|point| {
            union()
                .project(point)
                .is_some_and(|projection| !projection.exact)
        })
        .collect();
    assert!(!inexact.is_empty());
    inexact.iter().for_each(|point| {
        assert!(!difference.project(point).unwrap().exact);
    })
}

#[test]
fn zz_debug_nested() {
    use crate::geometry::csg::ops::test::grid;
    let (outer, inner) = (solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let oracle = Difference::new(solid([0.0; 3], [1.0; 3]), solid([0.3; 3], [0.7; 3]));
    let coordinates = |p: &Coordinate<3>| [p[0].value(), p[1].value(), p[2].value()];
    let mut missing = 0;
    let mut log = String::new();
    for point in grid().iter() {
        if oracle.project(point).is_some() {
            continue;
        }
        missing += 1;
        if missing > 6 {
            continue;
        }
        let outer_projection = outer.project(point).map(|q| q.point);
        let inner_projection = inner.project(point).map(|q| q.point);
        let describe = |q: &Option<Coordinate<3>>| {
            q.as_ref().map(|q| {
                (
                    coordinates(q),
                    outer.signed_distance(q).value(),
                    inner.signed_distance(q).value(),
                )
            })
        };
        log += &format!(
            "point {:?}\n  outer {:?}\n  inner {:?}\n",
            coordinates(point),
            describe(&outer_projection),
            describe(&inner_projection)
        );
    }
    panic!("missing {missing}\n{log}");
}
