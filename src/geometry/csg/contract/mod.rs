//! What every solid known through its signed distance must satisfy, as checks over many points.

use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::{
        CrossProduct, Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};
use core::{array::from_fn, f64::consts::TAU};

/// Points on a grid across a box, with a margin around it so that points inside, outside and near
/// the surface are all there.
pub fn grid(low: [f64; 3], high: [f64; 3], count: usize) -> Vec<Coordinate<3>> {
    let spacing = |axis: usize| (high[axis] - low[axis]) / (count - 1) as f64;
    let at = |axis: usize, index: usize| {
        let margin = 0.35 * (high[axis] - low[axis]);
        low[axis] - margin
            + (high[axis] - low[axis] + 2.0 * margin) * index as f64 / (count - 1) as f64
            + 0.013 * spacing(axis)
    };
    let mut points = Vec::new();
    for i in 0..count {
        for j in 0..count {
            for k in 0..count {
                points.push([at(0, i), at(1, j), at(2, k)].into());
            }
        }
    }
    points
}

/// Points along a polyline of `[distance from the axis, distance along it]` corners, `count` to a
/// segment.
pub fn along_polyline(
    corners: &[[Quantity<Length>; 2]],
    count: usize,
) -> Vec<[Quantity<Length>; 2]> {
    let mut points = Vec::new();
    for pair in corners.windows(2) {
        for i in 0..count {
            let t = i as f64 / count as f64;
            points.push(from_fn(|k| pair[0][k] + (pair[1][k] - pair[0][k]) * t));
        }
    }
    points.push(*corners.last().unwrap());
    points
}

/// How far along a unit `axis` through `origin` a point is, and how far from it.
pub fn local(
    origin: &Coordinate<3>,
    axis: &Direction<3>,
    point: &Coordinate<3>,
) -> (Quantity<Length>, Quantity<Length>) {
    let offset = point - origin;
    let along = &offset * axis;
    let across = (&offset - &(axis * along)).norm();
    (along, across)
}

/// The points of a profile turned `count` times about the axis through `origin` along the unit
/// vector `axis`.
pub fn revolved(
    origin: &Coordinate<3>,
    axis: &Direction<3>,
    profile: &[[Quantity<Length>; 2]],
    count: usize,
) -> Vec<Coordinate<3>> {
    let seed = if axis[0].abs() < 0.9 {
        Direction::const_from([1.0, 0.0, 0.0])
    } else {
        Direction::const_from([0.0, 1.0, 0.0])
    };
    let first = axis.cross(&seed).normalized();
    let second = axis.cross(&first);
    let mut points = Vec::new();
    for &[distance, along] in profile {
        for i in 0..count {
            let angle = TAU * i as f64 / count as f64;
            let outward = &(&first * angle.cos()) + &(&second * angle.sin());
            points.push(&(origin + &(axis * along)) + &(&outward * distance));
        }
    }
    points
}

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

/// A failure, with the point it was found at.
fn at(point: &Coordinate<3>, result: Result<(), AssertionError>) -> Result<(), AssertionError> {
    result.map_err(|error| AssertionError {
        message: format!("\n{point:?}{}", error.message),
    })
}

fn failure(message: String) -> Result<(), AssertionError> {
    Err(AssertionError { message })
}

/// The projection lies on the surface at the distance the signed distance says, with a unit
/// normal that points away from the solid, and the sign of the signed distance is `inside`'s.
///
/// `scale` is a length that the solid is measured in, which the tolerances are relative to.
pub fn assert_contract(
    solid: &impl SolidOracle,
    points: &[Coordinate<3>],
    scale: f64,
    inside: impl Fn(&Coordinate<3>) -> bool,
) -> Result<(), AssertionError> {
    let tols = Assert {
        abs_tol: 1e-9 * scale,
        rel_tol: 1e-9,
        ..Assert::default()
    };
    let slack = length(1e-9 * scale);
    for point in points {
        let check = || -> Result<(), AssertionError> {
            let distance = solid.signed_distance(point);
            let projection = solid.project(point).unwrap();
            let reach = (&projection.point - point).norm();
            tols.eq_within_tols(reach, &distance.abs())?;
            tols.zero_within_tols(&solid.signed_distance(&projection.point))?;
            tols.eq_within_tols(projection.normal.norm(), &Quantity::new(1.0))?;
            let along = &(point - &projection.point) * &projection.normal;
            if distance > slack {
                Assert::non_negative(&(along + slack))?;
            }
            if distance < -slack {
                Assert::non_positive(&(along - slack))?;
            }
            if distance.abs() > slack && (distance < Quantity::default()) != inside(point) {
                failure(format!(
                    "the sign of the signed distance {distance} is wrong"
                ))?;
            }
            if !projection.exact {
                failure("the projection is not exact".to_string())?;
            }
            Ok(())
        };
        at(point, check())?;
    }
    Ok(())
}

/// A distance field cannot change faster than the distance moved.
pub fn assert_lipschitz(
    solid: &impl SolidOracle,
    points: &[Coordinate<3>],
    scale: f64,
) -> Result<(), AssertionError> {
    let distances: Vec<Quantity<Length>> = points
        .iter()
        .map(|point| solid.signed_distance(point))
        .collect();
    let slack = length(1e-9 * scale);
    for (i, a) in points.iter().enumerate() {
        for (j, b) in points.iter().enumerate().skip(i + 1).step_by(7) {
            let change = (distances[i] - distances[j]).abs();
            let moved = (a - b).norm();
            at(
                a,
                Assert::non_positive(&(change - moved - slack)).map_err(|error| AssertionError {
                    message: format!(
                        "\nthe distance changes faster than the points move, to {b:?}{}",
                        error.message
                    ),
                }),
            )?;
        }
    }
    Ok(())
}

/// The signed distance is the distance to the surface, which `samples` of it, `spacing` apart at
/// most, bound from above, and from below to within a spacing.
pub fn assert_against_samples(
    solid: &impl SolidOracle,
    points: &[Coordinate<3>],
    samples: &[Coordinate<3>],
    spacing: f64,
) -> Result<(), AssertionError> {
    let slack = length(1e-9 * spacing);
    for point in points {
        let distance = solid.signed_distance(point).abs();
        let nearest = samples
            .iter()
            .map(|sample| (sample - point).norm())
            .min_by(|a, b| a.total_cmp(b))
            .unwrap();
        at(point, Assert::non_positive(&(distance - nearest - slack)))?;
        at(
            point,
            Assert::non_positive(&(nearest - distance - length(spacing))),
        )?;
    }
    Ok(())
}
