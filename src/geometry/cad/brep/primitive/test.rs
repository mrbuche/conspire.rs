use crate::geometry::{
    Coordinate,
    cad::brep::test::{axis_aligned_box, ball, capped_cylinder, cone, coplanar_squares, torus},
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
fn recognises_a_sphere() {
    let Some(Primitive::Sphere(sphere)) = ball(3.0).primitive() else {
        panic!("sphere not recognised");
    };
    let oracle = sphere.oracle().unwrap();
    assert!((oracle.signed_distance(&point([0.0; 3])) - 3.0).abs() < 1e-9);
    assert!(oracle.signed_distance(&point([3.0, 0.0, 0.0])).abs() < 1e-9);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 5.0])) < 0.0);
}

#[test]
fn recognises_a_truncated_cone() {
    let Some(Primitive::Cone(cone)) = cone(2.0, 1.0, 4.0).primitive() else {
        panic!("cone not recognised");
    };
    let oracle = cone.oracle().unwrap();
    // The recognised frustum spans the same rims: r = 2 at z = 0, r = 1 at z = 4.
    assert!(oracle.signed_distance(&point([2.0, 0.0, 0.0])).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([1.0, 0.0, 4.0])).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 2.0])) > 0.0);
    assert!((oracle.signed_distance(&point([0.0, 0.0, -3.0])) + 3.0).abs() < 1e-6);
}

#[test]
fn recognises_a_ring_torus() {
    let Some(Primitive::Torus(ring)) = torus(3.0, 1.0).primitive() else {
        panic!("torus not recognised");
    };
    let oracle = ring.oracle().unwrap();
    assert!((oracle.signed_distance(&point([3.0, 0.0, 0.0])) - 1.0).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([4.0, 0.0, 0.0])).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([0.0; 3])) < 0.0);
}

#[test]
fn declines_a_non_primitive() {
    // An open two-face shell reduces to nothing.
    assert!(coplanar_squares().primitive().is_none());
}
