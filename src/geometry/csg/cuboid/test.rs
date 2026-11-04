use super::Cuboid;
use crate::{
    geometry::{
        Coordinate,
        csg::contract::{assert_against_samples, assert_contract, assert_lipschitz, grid},
        solid::SolidOracle,
    },
    math::{Quantity, Tensor},
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn corner(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

#[test]
fn signed_distance_is_negative_inside_and_euclidean_past_a_corner() {
    let cuboid = Cuboid::new(corner([0.0; 3]), corner([2.0, 4.0, 8.0])).unwrap();
    let distance = |entries| cuboid.signed_distance(&corner(entries));
    assert!((distance([1.0, 2.0, 4.0]) + length(1.0)).abs() < length(1e-12));
    assert!((distance([5.0, 2.0, 4.0]) - length(3.0)).abs() < length(1e-12));
    assert!(distance([0.0, 2.0, 4.0]).abs() < length(1e-12));
    let expected = (1.0_f64 + 4.0 + 4.0).sqrt();
    assert!((distance([3.0, 6.0, 10.0]) - length(expected)).abs() < length(1e-12));
}

#[test]
fn projects_onto_the_nearest_face_or_clamps_onto_the_surface() {
    let cuboid = Cuboid::new(corner([0.0; 3]), corner([2.0; 3])).unwrap();
    let inside = cuboid.project(&corner([1.9, 1.0, 1.0])).unwrap();
    assert!((&inside.point - &corner([2.0, 1.0, 1.0])).norm() < length(1e-12));
    assert!((&inside.normal - &[1.0, 0.0, 0.0].into()).norm() < Quantity::new(1e-12));
    let outside = cuboid.project(&corner([3.0, 1.0, 1.0])).unwrap();
    assert!((&outside.point - &corner([2.0, 1.0, 1.0])).norm() < length(1e-12));
    let past = cuboid.project(&corner([3.0, 3.0, 2.0])).unwrap();
    assert!((&past.point - &corner([2.0, 2.0, 2.0])).norm() < length(1e-12));
    let diagonal = 0.5_f64.sqrt();
    assert!((&past.normal - &[diagonal, diagonal, 0.0].into()).norm() < Quantity::new(1e-12));
}

#[test]
fn a_box_with_no_extent_is_an_error() {
    assert!(Cuboid::new(corner([0.0; 3]), corner([1.0, 1.0, 0.0])).is_err());
    assert!(Cuboid::new(corner([0.0; 3]), corner([1.0, -1.0, 1.0])).is_err());
}

fn faces(low: [f64; 3], high: [f64; 3], count: usize) -> Vec<Coordinate<3>> {
    let at = |axis: usize, index: usize| {
        low[axis] + (high[axis] - low[axis]) * index as f64 / (count - 1) as f64
    };
    let mut samples = Vec::new();
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for end in [low[axis], high[axis]] {
            for i in 0..count {
                for j in 0..count {
                    let mut sample = [0.0; 3];
                    sample[axis] = end;
                    sample[u] = at(u, i);
                    sample[v] = at(v, j);
                    samples.push(corner(sample));
                }
            }
        }
    }
    samples
}

#[test]
fn satisfies_the_contract_at_any_size_and_place() {
    for (low, high) in [
        ([0.0, 0.0, 0.0], [2.0, 4.0, 8.0]),
        ([-1.0, 2.0, 0.5], [0.5, 2.1, 3.0]),
        ([0.0; 3], [1e-9, 2e-9, 3e-9]),
        ([1e8, -1e8, 0.0], [3e8, 1e8, 1e8]),
    ] {
        let cuboid = Cuboid::new(corner(low), corner(high)).unwrap();
        let scale = (0..3)
            .map(|axis| high[axis] - low[axis])
            .fold(f64::INFINITY, f64::min);
        let points = grid(low, high, 7);
        let inside = |p: &Coordinate<3>| {
            (0..3).all(|axis| low[axis] < p[axis].value() && p[axis].value() < high[axis])
        };
        assert_contract(&cuboid, &points, scale, inside);
        assert_lipschitz(&cuboid, &points, scale);
    }
}

#[test]
fn agrees_with_a_sampling_of_its_surface() {
    let (low, high) = ([0.0, -1.0, 0.5], [2.0, 1.0, 1.5]);
    let cuboid = Cuboid::new(corner(low), corner(high)).unwrap();
    let samples = faces(low, high, 41);
    let spacing = (2.0_f64).sqrt() * 2.0 / 40.0 * 0.5 + 1e-12;
    assert_against_samples(&cuboid, &grid(low, high, 6), &samples, spacing);
}
