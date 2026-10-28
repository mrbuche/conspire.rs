use crate::{
    geometry::{
        Coordinate, Direction,
        mesh::{Tessellation, test::mesh},
        solid::SolidOracle,
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

fn cube() -> Tessellation {
    mesh().into()
}

fn box_distance(point: [f64; 3]) -> Quantity<Length> {
    let q = point.map(|x| (x - 0.5).abs() - 0.5);
    let outside = q
        .map(|x| x.max(0.0))
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    Length::meters(outside + q.iter().cloned().fold(f64::NEG_INFINITY, f64::max).min(0.0))
}

const POINTS: [[f64; 3]; 9] = [
    [0.5, 0.5, 0.5],
    [0.2, 0.4, 0.6],
    [0.9, 0.5, 0.3],
    [1.5, 0.5, 0.5],
    [-0.3, 0.5, 0.5],
    [0.5, 2.0, 0.5],
    [1.5, 1.5, 1.5],
    [-0.2, 1.3, 0.4],
    [0.5, 0.5, -0.7],
];

#[test]
fn signed_distance_matches_box() -> Result<(), AssertionError> {
    let cube = cube();
    POINTS.iter().try_for_each(|&point| {
        Assert::default().eq_within_tols(cube.signed_distance(&point.into()), &box_distance(point))
    })
}

#[test]
fn project_lands_on_the_boundary_at_the_distance() -> Result<(), AssertionError> {
    let cube = cube();
    POINTS.iter().try_for_each(|&point| {
        let point = Coordinate::from(point);
        let projection = cube.project(&point).unwrap();
        assert!(projection.exact);
        let projected = projection.point;
        Assert::default().eq_within_tols(cube.signed_distance(&projected), &Length::meters(0.0))?;
        Assert::default().eq_within_tols(
            (&projected - &point).norm(),
            &cube.signed_distance(&point).abs(),
        )
    })
}

#[test]
fn sign_is_negative_inside_and_positive_outside() -> Result<(), AssertionError> {
    let cube = cube();
    Assert::non_positive(&cube.signed_distance(&[0.5; 3].into()))?;
    Assert::non_negative(&cube.signed_distance(&[2.0; 3].into()))
}

#[test]
fn projection_normal_points_out_of_the_solid() -> Result<(), AssertionError> {
    let cube = cube();
    [
        ([2.0, 0.5, 0.5], [1.0, 0.0, 0.0]),
        ([-1.0, 0.5, 0.5], [-1.0, 0.0, 0.0]),
        ([0.5, 2.0, 0.5], [0.0, 1.0, 0.0]),
        ([0.5, -1.0, 0.5], [0.0, -1.0, 0.0]),
        ([0.5, 0.5, 2.0], [0.0, 0.0, 1.0]),
        ([0.5, 0.5, 0.9], [0.0, 0.0, 1.0]),
        ([0.1, 0.5, 0.5], [-1.0, 0.0, 0.0]),
    ]
    .iter()
    .try_for_each(|&(point, normal)| {
        let projected = cube.project(&point.into()).unwrap().normal;
        Assert::default().eq_within_tols(&projected, &Direction::from(normal))
    })
}
