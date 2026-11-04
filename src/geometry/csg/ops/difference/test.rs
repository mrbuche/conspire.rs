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
    let mut inexact = 0;
    let mut log = String::new();
    for point in grid().iter() {
        let Some(result) = oracle.project(point) else {
            continue;
        };
        if result.exact {
            continue;
        }
        inexact += 1;
        if inexact > 4 {
            continue;
        }
        let describe = |name: &str, q: Option<crate::geometry::solid::Projection>| {
            q.map(|q| {
                format!(
                    "  {name} point {:?} exact {} sd_outer {:e} sd_inner {:e} dist {:e}\n",
                    coordinates(&q.point),
                    q.exact,
                    outer.signed_distance(&q.point).value(),
                    inner.signed_distance(&q.point).value(),
                    (&q.point - point).norm().value(),
                )
            })
            .unwrap_or_default()
        };
        log += &format!("query {:?}\n", coordinates(point));
        log += &describe("outer", outer.project(point));
        log += &describe("inner", inner.project(point));
        log += &describe("result", Some(result));
    }
    panic!("inexact {inexact}\n{log}");
}
