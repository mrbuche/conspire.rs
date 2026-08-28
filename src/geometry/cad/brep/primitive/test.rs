use crate::geometry::{
    Coordinate,
    cad::brep::test::{axis_aligned_box, capped_cylinder, coplanar_squares},
    csg::Primitive,
    solid::{Solid, SolidOracle},
};

fn point(entries: [f64; 3]) -> Coordinate<3> {
    Coordinate::from(entries)
}

#[test]
fn recognises_an_axis_aligned_box() {
    let Some(Primitive::Cuboid(cuboid)) = axis_aligned_box([2.0, 4.0, 8.0]).primitive() else {
        panic!("box not recognised");
    };
    let oracle = cuboid.oracle().unwrap();
    // Centre: distance to the nearest face is the smallest half-extent.
    assert!((oracle.signed_distance(&point([1.0, 2.0, 4.0])) - 1.0).abs() < 1e-9);
    // On the -x face.
    assert!(oracle.signed_distance(&point([0.0, 2.0, 4.0])).abs() < 1e-9);
    // Outside.
    assert!(oracle.signed_distance(&point([3.0, 2.0, 4.0])) < 0.0);
}

#[test]
fn recognises_a_capped_cylinder() {
    let Some(Primitive::Cylinder(cylinder)) = capped_cylinder(2.0, 5.0).primitive() else {
        panic!("cylinder not recognised");
    };
    let oracle = cylinder.oracle().unwrap();
    // On the axis, mid-height: nearest surface is the lateral wall, 2 away.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 2.5])) - 2.0).abs() < 1e-9);
    // On the wall, and on the base cap.
    assert!(oracle.signed_distance(&point([2.0, 0.0, 2.5])).abs() < 1e-9);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 0.0])).abs() < 1e-9);
    // Past the top cap.
    assert!(oracle.signed_distance(&point([0.0, 0.0, 6.0])) < 0.0);
}

#[test]
fn declines_a_non_primitive() {
    // An open two-face shell reduces to nothing.
    assert!(coplanar_squares().primitive().is_none());
}
