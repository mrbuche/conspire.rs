use crate::{
    geometry::{
        Coordinate, Direction,
        csg::Cone,
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

fn frustum() -> Cone {
    // Base radius 2 at z = 0, tip radius 1 at z = 4.
    Cone::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 2.0, 1.0, 4.0).unwrap()
}

#[test]
fn rejects_bad_dimensions() {
    assert!(Cone::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 0.0, 0.0, 1.0).is_err());
    assert!(Cone::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 1.0, 1.0, 0.0).is_err());
    assert!(Cone::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), -1.0, 1.0, 1.0).is_err());
}

#[test]
fn signed_distance_on_caps_and_outside() {
    let oracle = frustum().oracle().unwrap();
    // On the base rim and the tip rim.
    assert!(oracle.signed_distance(&point([2.0, 0.0, 0.0])).abs() < 1e-9);
    assert!(oracle.signed_distance(&point([1.0, 0.0, 4.0])).abs() < 1e-9);
    // Just inside each cap, on the axis.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 0.5])) - 0.5).abs() < 1e-9);
    assert!((oracle.signed_distance(&point([0.0, 0.0, 3.9])) - 0.1).abs() < 1e-9);
    // Three below the base plane, and one past the base rim.
    assert!((oracle.signed_distance(&point([0.0, 0.0, -3.0])) + 3.0).abs() < 1e-9);
    assert!((oracle.signed_distance(&point([3.0, 0.0, 0.0])) + 1.0).abs() < 1e-9);
}

#[test]
fn projects_onto_the_lateral_wall() {
    let oracle = frustum().oracle().unwrap();
    let (surface, normal) = oracle.project(&point([3.0, 0.0, 2.0])).unwrap();
    // Foot of the perpendicular from (3, 2) onto the line (2, 0) -> (1, 4).
    assert!((surface[0].value() - 1.588).abs() < 5e-3);
    assert!((surface[2].value() - 1.647).abs() < 5e-3);
    // Outward normal leans out and slightly up.
    assert!(normal[0].value() > 0.9 && normal[2].value() > 0.2);
}

#[test]
fn meshes_the_frustum() {
    let mesh = frustum()
        .mesh(
            &Uniform(length(0.5)),
            6,
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
    assert!((low[0] + 2.0).abs() < 0.3 && (high[0] - 2.0).abs() < 0.3);
    assert!(low[2].abs() < 0.15 && (high[2] - 4.0).abs() < 0.15);
}
