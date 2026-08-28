use crate::{
    geometry::{
        Coordinate,
        csg::Cuboid,
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

fn corner(entries: [f64; 3]) -> Coordinate<3> {
    Coordinate::from(entries)
}

#[test]
fn signed_distance_sign_and_magnitude() {
    let cuboid = Cuboid::new(corner([0.0; 3]), corner([2.0, 4.0, 8.0])).unwrap();
    let oracle = cuboid.oracle().unwrap();
    // Centre: distance to the nearest face is the smallest half-extent.
    assert!((oracle.signed_distance(&corner([1.0, 2.0, 4.0])) - 1.0).abs() < 1e-12);
    // Three units past the +x face, negative outside.
    assert!((oracle.signed_distance(&corner([5.0, 2.0, 4.0])) + 3.0).abs() < 1e-12);
    // On a face.
    assert!(oracle.signed_distance(&corner([0.0, 2.0, 4.0])).abs() < 1e-12);
    // Past a corner: Euclidean distance to it.
    let expected = (1.0f64 + 4.0 + 4.0).sqrt();
    assert!((oracle.signed_distance(&corner([3.0, 6.0, 10.0])) + expected).abs() < 1e-12);
}

#[test]
fn projects_onto_the_nearest_face() {
    let cuboid = Cuboid::new(corner([0.0; 3]), corner([2.0, 2.0, 2.0])).unwrap();
    let oracle = cuboid.oracle().unwrap();

    let (point, normal) = oracle.project(&corner([1.9, 1.0, 1.0])).unwrap();
    assert!((point[0].value() - 2.0).abs() < 1e-12);
    assert!((point[1].value() - 1.0).abs() < 1e-12);
    assert!((normal[0].value() - 1.0).abs() < 1e-12);

    // An exterior point clamps onto the box surface.
    let (point, _) = oracle.project(&corner([3.0, 1.0, 1.0])).unwrap();
    assert!((point[0].value() - 2.0).abs() < 1e-12);
}

#[test]
fn meshes_the_overhanging_box() {
    // The octree root is a cube of the longest side, so classify + trim must
    // carve the block back to the 2x4x8 geometry on the two short axes.
    let cuboid = Cuboid::new(corner([0.0; 3]), corner([2.0, 4.0, 8.0])).unwrap();
    let mesh = cuboid
        .mesh(
            &Uniform(length(1.0)),
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

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    let extents = [2.0, 4.0, 8.0];
    for axis in 0..3 {
        assert!(low[axis].abs() < 5e-2, "low[{axis}] = {}", low[axis]);
        assert!(
            (high[axis] - extents[axis]).abs() < 5e-2,
            "high[{axis}] = {}",
            high[axis]
        );
    }
}
