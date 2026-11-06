use super::Sphere;
use crate::{
    geometry::{
        Coordinate, Direction,
        csg::contract::{assert_against_samples, assert_contract, assert_lipschitz, grid},
        solid::SolidOracle,
    },
    math::{
        Quantity, Tensor,
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

#[test]
fn signed_distance_is_radial_and_negative_inside() -> Result<(), AssertionError> {
    let sphere = Sphere::new(point([1.0, 2.0, 3.0]), length(4.0)).unwrap();
    let distance = |entries| sphere.signed_distance(&point(entries));
    let assert = Assert::default();
    assert.eq_within_tols(distance([1.0, 2.0, 3.0]), &length(-4.0))?;
    assert.eq_within_tols(distance([1.0, 2.0, 9.0]), &length(2.0))?;
    assert.zero_within_tols(&distance([5.0, 2.0, 3.0]))
}

#[test]
fn projects_along_the_radius() -> Result<(), AssertionError> {
    let sphere = Sphere::new(point([0.0; 3]), length(3.0)).unwrap();
    let projection = sphere.project(&point([6.0, 0.0, 0.0])).unwrap();
    let assert = Assert::default();
    assert.eq_within_tols(&projection.point, &point([3.0, 0.0, 0.0]))?;
    assert.eq_within_tols(&projection.normal, &Direction::from([1.0, 0.0, 0.0]))
}

#[test]
fn the_centre_projects_somewhere_on_the_surface() -> Result<(), AssertionError> {
    let sphere = Sphere::new(point([1.0, 2.0, 3.0]), length(2.0)).unwrap();
    let projection = sphere.project(&point([1.0, 2.0, 3.0])).unwrap();
    let radius = (&projection.point - &point([1.0, 2.0, 3.0])).norm();
    Assert::default().eq_within_tols(radius, &length(2.0))
}

#[test]
fn a_radius_that_is_not_positive_is_an_error() {
    assert!(Sphere::new(point([0.0; 3]), length(0.0)).is_err());
    assert!(Sphere::new(point([0.0; 3]), length(-1.0)).is_err());
}

fn surface(center: [f64; 3], radius: f64, count: usize) -> Vec<Coordinate<3>> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let ring = (1.0 - z * z).sqrt();
            let angle = golden * i as f64;
            point([
                center[0] + radius * ring * angle.cos(),
                center[1] + radius * ring * angle.sin(),
                center[2] + radius * z,
            ])
        })
        .collect()
}

#[test]
fn satisfies_the_contract_at_any_size_and_place() -> Result<(), AssertionError> {
    for (center, radius) in [
        ([1.0, -2.0, 0.5], 1.5),
        ([0.0; 3], 1e-9),
        ([5e8, 0.0, 0.0], 3e8),
        ([1e6, 1e6, 1e6], 1.0),
    ] {
        let sphere = Sphere::new(point(center), length(radius)).unwrap();
        let low = center.map(|x| x - radius);
        let high = center.map(|x| x + radius);
        let points = grid(low, high, 7);
        let inside = |p: &Coordinate<3>| (p - &point(center)).norm() < length(radius);
        assert_contract(&sphere, &points, radius, inside)?;
        assert_lipschitz(&sphere, &points, radius)?;
    }
    Ok(())
}

#[test]
fn agrees_with_a_sampling_of_its_surface() -> Result<(), AssertionError> {
    let (center, radius) = ([0.3, -0.2, 0.1], 1.0);
    let sphere = Sphere::new(point(center), length(radius)).unwrap();
    let samples = surface(center, radius, 4000);
    let spacing = (4.0 * PI / 4000.0_f64).sqrt() * 1.3;
    let points = grid(center.map(|x| x - radius), center.map(|x| x + radius), 6);
    assert_against_samples(&sphere, &points, &samples, spacing)
}
