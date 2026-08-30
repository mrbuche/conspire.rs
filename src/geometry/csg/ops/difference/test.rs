use crate::{
    geometry::{
        Coordinate,
        csg::{Cuboid, Sphere, ops::Difference},
        mesh::{Connectivity, Fitting, Verdict},
        ntree::Balancing,
        solid::{Solid, SolidOracle, Uniform},
    },
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    Coordinate::from(entries)
}

fn porous_block() -> Difference<Cuboid, Sphere> {
    Difference::new(
        Cuboid::new(point([-3.0; 3]), point([3.0; 3])).unwrap(),
        Sphere::new(point([0.0; 3]), 1.5).unwrap(),
    )
}

fn radius_from_origin(coordinate: &Coordinate<3>) -> f64 {
    (0..3)
        .map(|k| coordinate[k].value().powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn signed_distance_carves_the_pore() {
    let oracle = porous_block().oracle().unwrap();
    // Pore centre: deepest point of the removed region, 1.5 outside the solid.
    assert!((oracle.signed_distance(&point([0.0; 3])) + 1.5).abs() < 1e-12);
    // Through solid material between pore and wall: positive.
    assert!(oracle.signed_distance(&point([2.25, 0.0, 0.0])) > 0.0);
    // On the pore wall, and on the outer wall.
    assert!(oracle.signed_distance(&point([1.5, 0.0, 0.0])).abs() < 1e-12);
    assert!(oracle.signed_distance(&point([3.0, 0.0, 0.0])).abs() < 1e-12);
}

#[test]
fn meshes_a_porous_block() {
    let mesh = porous_block()
        .mesh(
            &Uniform(length(0.6)),
            Some(6),
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
        )
        .unwrap();

    assert_eq!(mesh.connectivities().len(), 1);
    assert!(matches!(
        mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    let jacobians = mesh.minimum_scaled_jacobians();
    assert!(
        jacobians[0].iter().all(|&j| j > 0.0),
        "inverted hex: worst scaled Jacobian {}",
        jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min)
    );

    // The pore is hollow: no node sits well inside the sphere, and the fit
    // brings the cavity wall out toward its radius.
    let mut closest = f64::INFINITY;
    for coordinate in mesh.coordinates() {
        closest = closest.min(radius_from_origin(coordinate));
    }
    assert!(closest > 1.5 - 0.25, "a node intruded the pore at r = {closest}");
    assert!(closest < 1.5 + 0.35, "cavity wall never reached, r = {closest}");

    // The outer block is still the box.
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    for (k, &value) in high.iter().enumerate() {
        assert!((value - 3.0).abs() < 5e-2, "high[{k}] = {value}");
    }
}
