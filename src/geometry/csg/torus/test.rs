use crate::{
    geometry::{
        Coordinate, Direction,
        csg::Torus,
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

fn axis(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

fn ring() -> Torus {
    // Major radius 3, tube radius 1, about +z.
    Torus::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 3.0, 1.0).unwrap()
}

#[test]
fn rejects_a_self_crossing_torus() {
    assert!(Torus::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 1.0, 1.0).is_err());
    assert!(Torus::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 3.0, 0.0).is_err());
}

#[test]
fn signed_distance_is_the_tube() {
    let oracle = ring().oracle().unwrap();
    // The tube centre circle: distance to the surface is the tube radius.
    assert!((oracle.signed_distance(&point([3.0, 0.0, 0.0])) - 1.0).abs() < 1e-12);
    // Outer and inner equators, and the top of the tube: on the surface.
    assert!(oracle.signed_distance(&point([4.0, 0.0, 0.0])).abs() < 1e-12);
    assert!(oracle.signed_distance(&point([2.0, 0.0, 0.0])).abs() < 1e-12);
    assert!(oracle.signed_distance(&point([3.0, 0.0, 1.0])).abs() < 1e-12);
    // The hole centre and the axis are outside.
    assert!(oracle.signed_distance(&point([0.0; 3])) < 0.0);
    assert!((oracle.signed_distance(&point([6.0, 0.0, 0.0])) + 2.0).abs() < 1e-12);
}

#[test]
fn projects_onto_the_tube() {
    let oracle = ring().oracle().unwrap();
    let (surface, normal) = oracle.project(&point([5.0, 0.0, 0.0])).unwrap();
    assert!((surface[0].value() - 4.0).abs() < 1e-12);
    assert!(surface[1].value().abs() < 1e-12 && surface[2].value().abs() < 1e-12);
    assert!((normal[0].value() - 1.0).abs() < 1e-12);
}

#[test]
fn meshes_the_ring() {
    let mesh = ring()
        .mesh(
            &Uniform(length(0.4)),
            Some(6),
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
        )
        .unwrap();
    assert!(matches!(
        mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    assert!(mesh.minimum_scaled_jacobians()[0].iter().all(|&j| j > 0.0));

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    // Outer diameter 8, tube height 2.
    assert!((high[0] - 4.0).abs() < 0.25 && (low[0] + 4.0).abs() < 0.25);
    assert!((high[2] - 1.0).abs() < 0.25 && (low[2] + 1.0).abs() < 0.25);
}
