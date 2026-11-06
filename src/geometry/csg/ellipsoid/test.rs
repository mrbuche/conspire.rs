use super::Ellipsoid;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::contract::{assert_against_samples, assert_contract, assert_lipschitz, grid},
        solid::SolidOracle,
    },
    math::{
        CrossProduct, Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};
use core::f64::consts::PI;

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn direction(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

fn semi(entries: [f64; 3]) -> [Quantity<Length>; 3] {
    entries.map(length)
}

fn frame(first: [f64; 3]) -> [Direction<3>; 3] {
    let a = direction(first).normalized();
    let seed = if a[0].abs() < 0.9 {
        direction([1.0, 0.0, 0.0])
    } else {
        direction([0.0, 1.0, 0.0])
    };
    let b = a.cross(&seed).normalized();
    let c = a.cross(&b);
    [a, b, c]
}

#[test]
fn along_an_axis_the_distance_is_to_its_tip() -> Result<(), AssertionError> {
    let ellipsoid = Ellipsoid::new(point([1.0, 2.0, 3.0]), semi([4.0, 2.0, 1.0])).unwrap();
    let distance = |entries| ellipsoid.signed_distance(&point(entries));
    let assert = Assert::default();
    assert.eq_within_tols(distance([6.0, 2.0, 3.0]), &length(1.0))?;
    assert.eq_within_tols(distance([1.0, 2.0, 5.0]), &length(1.0))?;
    assert.zero_within_tols(&distance([5.0, 2.0, 3.0]))?;
    assert.eq_within_tols(distance([1.0, 2.0, 3.0]), &length(-1.0))
}

#[test]
fn a_ball_is_a_sphere() -> Result<(), AssertionError> {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), semi([2.0, 2.0, 2.0])).unwrap();
    for entries in [
        [1.0, 0.0, 0.0],
        [0.3, -1.1, 4.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ] {
        let p = point(entries);
        let expected = p.norm() - length(2.0);
        Assert::default().eq_within_tols(ellipsoid.signed_distance(&p), &expected)?;
    }
    Ok(())
}

#[test]
fn a_point_in_a_plane_of_a_flat_ellipsoid_is_nearest_a_point_out_of_it()
-> Result<(), AssertionError> {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), semi([2.0, 2.0, 0.5])).unwrap();
    let found = ellipsoid.project(&point([0.1, 0.0, 0.0])).unwrap();
    Assert::non_negative(&(found.point[2].abs() - length(0.4)))?;
    Assert::non_positive(
        &(ellipsoid.signed_distance(&point([0.1, 0.0, 0.0])).abs() - length(0.51)),
    )?;
    let rim = ellipsoid.project(&point([1.9, 0.0, 0.0])).unwrap();
    Assert::default().zero_within_tols(&rim.point[2])
}

#[test]
fn an_ellipsoid_that_is_not_one_is_an_error() {
    let centre = point([0.0; 3]);
    assert!(Ellipsoid::new(centre.clone(), semi([1.0, 2.0, 3.0])).is_ok());
    assert!(Ellipsoid::new(centre.clone(), semi([1.0, 0.0, 3.0])).is_err());
    assert!(Ellipsoid::new(centre.clone(), semi([1.0, -2.0, 3.0])).is_err());
    let slanted = [
        direction([1.0, 0.0, 0.0]),
        direction([0.5, 0.5_f64.sqrt() * 1.2, 0.0]),
        direction([0.0, 0.0, 1.0]),
    ];
    assert!(Ellipsoid::oriented(centre.clone(), slanted, semi([1.0, 2.0, 3.0])).is_err());
    let long = [
        direction([2.0, 0.0, 0.0]),
        direction([0.0, 1.0, 0.0]),
        direction([0.0, 0.0, 1.0]),
    ];
    assert!(Ellipsoid::oriented(centre.clone(), long, semi([1.0, 2.0, 3.0])).is_err());
    assert!(Ellipsoid::oriented(centre, frame([1.0, 2.0, 3.0]), semi([1.0, 2.0, 3.0])).is_ok());
}

type Case = ([f64; 3], [f64; 3], [f64; 3]);

const CASES: [Case; 8] = [
    ([0.0; 3], [1.0, 0.0, 0.0], [3.0, 2.0, 1.0]),
    ([1.0, -1.0, 0.5], [1.0, 2.0, 3.0], [2.0, 1.5, 0.7]),
    ([0.0; 3], [1.0, 1.0, 0.0], [2.0, 2.0, 0.05]),
    ([0.0; 3], [0.0, 0.0, 1.0], [3.0, 0.1, 0.1]),
    ([0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0001]),
    ([0.0; 3], [1.0, 0.0, 0.0], [3e-9, 2e-9, 1e-9]),
    ([4e8, 0.0, -1e8], [1.0, 2.0, 3.0], [3e8, 2e8, 1e8]),
    ([0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 2.0]),
];

fn make(case: Case) -> Ellipsoid {
    let (center, first, axes) = case;
    Ellipsoid::oriented(point(center), frame(first), semi(axes)).unwrap()
}

fn around(case: Case) -> ([f64; 3], [f64; 3]) {
    let (center, _, axes) = case;
    let reach = axes.into_iter().fold(0.0, f64::max);
    (center.map(|x| x - reach), center.map(|x| x + reach))
}

#[test]
fn satisfies_the_contract_for_any_size_place_and_shape() -> Result<(), AssertionError> {
    for case in CASES {
        let ellipsoid = make(case);
        let (center, first, axes) = case;
        let frame = frame(first);
        let (low, high) = around(case);
        let points = grid(low, high, 9);
        let scale = axes.into_iter().fold(f64::INFINITY, f64::min);
        let inside = |p: &Coordinate<3>| {
            let offset = p - &point(center);
            (0..3)
                .map(|i| ((&offset * &frame[i]) / length(axes[i])).value().powi(2))
                .sum::<f64>()
                < 1.0
        };
        assert_contract(&ellipsoid, &points, scale, inside)?;
        assert_lipschitz(&ellipsoid, &points, scale)?;
    }
    Ok(())
}

fn surface(case: Case, count: usize) -> Vec<Coordinate<3>> {
    let (center, first, axes) = case;
    let frame = frame(first);
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let ring = (1.0 - z * z).sqrt();
            let angle = golden * i as f64;
            let on_sphere = [ring * angle.cos(), ring * angle.sin(), z];
            (0..3).fold(point(center), |sum, k| {
                &sum + &(&frame[k] * length(axes[k] * on_sphere[k]))
            })
        })
        .collect()
}

#[test]
fn agrees_with_a_sampling_of_its_surface() -> Result<(), AssertionError> {
    let count = 40000;
    for case in &CASES[..5] {
        let ellipsoid = make(*case);
        let samples = surface(*case, count);
        let longest = case.2.into_iter().fold(0.0, f64::max);
        let spacing = 1.3 * longest * (4.0 * PI / count as f64).sqrt();
        let (low, high) = around(*case);
        assert_against_samples(&ellipsoid, &grid(low, high, 6), &samples, spacing)?;
    }
    Ok(())
}

#[test]
fn points_in_the_planes_and_on_the_axes_of_the_ellipsoid_are_as_good_as_any()
-> Result<(), AssertionError> {
    let count = 40000;
    for case in &CASES[..5] {
        let ellipsoid = make(*case);
        let (center, first, axes) = *case;
        let frame = frame(first);
        let samples = surface(*case, count);
        let longest = axes.into_iter().fold(0.0, f64::max);
        let spacing = 1.3 * longest * (4.0 * PI / count as f64).sqrt();
        let steps = [-1.4, -0.9, -0.3, -0.05, 0.0, 0.05, 0.3, 0.9, 1.4];
        let mut points = Vec::new();
        for &a in &steps {
            for &b in &steps {
                for plane in 0..3 {
                    let mut along = [0.0; 3];
                    along[(plane + 1) % 3] = a;
                    along[(plane + 2) % 3] = b;
                    points.push((0..3).fold(point(center), |sum, k| {
                        &sum + &(&frame[k] * length(axes[k] * along[k]))
                    }));
                }
            }
        }
        let scale = axes.into_iter().fold(f64::INFINITY, f64::min);
        let inside = |p: &Coordinate<3>| {
            let offset = p - &point(center);
            (0..3)
                .map(|i| ((&offset * &frame[i]) / length(axes[i])).value().powi(2))
                .sum::<f64>()
                < 1.0
        };
        assert_contract(&ellipsoid, &points, scale, inside)?;
        assert_against_samples(&ellipsoid, &points, &samples, spacing)?;
    }
    Ok(())
}
