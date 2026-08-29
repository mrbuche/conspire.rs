use crate::{
    geometry::{
        Coordinate, Direction,
        csg::Ellipsoid,
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

#[test]
fn rejects_a_non_positive_semi_axis() {
    assert!(Ellipsoid::new(point([0.0; 3]), [1.0, 0.0, 2.0]).is_err());
}

#[test]
fn signed_distance_axis_aligned() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [1.0, 2.0, 3.0]).unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    // Centre: nearest surface is the tip of the shortest semi-axis, 1 away.
    assert!((oracle.signed_distance(&point([0.0; 3])) - 1.0).abs() < 1e-6);
    // On the surface at each tip.
    assert!(oracle.signed_distance(&point([1.0, 0.0, 0.0])).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 3.0])).abs() < 1e-6);
    // Two units past the +x tip.
    assert!((oracle.signed_distance(&point([3.0, 0.0, 0.0])) + 2.0).abs() < 1e-6);
    // One unit past the +z tip.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 4.0])) + 1.0).abs() < 1e-6);
}

#[test]
fn projects_onto_the_surface() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [1.0, 2.0, 3.0]).unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    let (surface, normal) = oracle.project(&point([5.0, 0.0, 0.0])).unwrap();
    assert!((surface[0].value() - 1.0).abs() < 1e-6);
    assert!(surface[1].value().abs() < 1e-6 && surface[2].value().abs() < 1e-6);
    assert!((normal[0].value() - 1.0).abs() < 1e-6);
}

#[test]
fn oriented_swaps_the_axes() {
    // Rotate 90 degrees about z: the a=1 semi-axis now points along world +y.
    let ellipsoid = Ellipsoid::oriented(
        point([0.0; 3]),
        [
            axis([0.0, 1.0, 0.0]),
            axis([-1.0, 0.0, 0.0]),
            axis([0.0, 0.0, 1.0]),
        ],
        [1.0, 2.0, 3.0],
    )
    .unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    assert!(oracle.signed_distance(&point([0.0, 1.0, 0.0])).abs() < 1e-6);
    assert!((oracle.signed_distance(&point([0.0, 3.0, 0.0])) + 2.0).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([2.0, 0.0, 0.0])).abs() < 1e-6);
}

#[test]
fn meshes_the_ellipsoid() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [2.0, 3.0, 4.0]).unwrap();
    let mesh = ellipsoid
        .mesh(
            &Uniform(length(0.6)),
            6,
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

    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    let expected = [2.0, 3.0, 4.0];
    for (k, (&h, &e)) in high.iter().zip(&expected).enumerate() {
        assert!((h - e).abs() < 0.3, "high[{k}] = {h}");
    }
}
