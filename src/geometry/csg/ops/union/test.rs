use crate::{
    geometry::{
        Coordinate, Direction,
        csg::{
            Cuboid, Cylinder, Sphere,
            ops::{Difference, Union},
        },
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
fn signed_distance_of_heterogeneous_operands() {
    // A box with a cylindrical boss on its +z face -- no enum, no trait object.
    let solid = Union::new(
        Cuboid::new(point([-2.0; 3]), point([2.0; 3])).unwrap(),
        Cylinder::new(point([0.0, 0.0, 2.0]), axis([0.0, 0.0, 1.0]), 1.0, 1.5).unwrap(),
    );
    let oracle = solid.oracle().unwrap();
    assert!(oracle.signed_distance(&point([0.0; 3])) > 0.0);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 3.0])) > 0.0);
    assert!(oracle.signed_distance(&point([1.8, 0.0, 3.0])) < 0.0);
}

#[test]
fn nests_with_a_combinator() {
    // Union of "a block with a pore" (a Difference, not a primitive) with a boss.
    let porous = Difference::new(
        Cuboid::new(point([-2.0; 3]), point([2.0; 3])).unwrap(),
        Sphere::new(point([0.0; 3]), 1.0).unwrap(),
    );
    let boss = Cylinder::new(point([0.0, 0.0, 2.0]), axis([0.0, 0.0, 1.0]), 1.0, 1.5).unwrap();
    let mesh = Union::new(porous, boss)
        .mesh(
            &Uniform(length(0.4)),
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

    let mut nearest = f64::INFINITY;
    let mut top = f64::NEG_INFINITY;
    for coordinate in mesh.coordinates() {
        let radius = (0..3)
            .map(|k| coordinate[k].value().powi(2))
            .sum::<f64>()
            .sqrt();
        nearest = nearest.min(radius);
        top = top.max(coordinate[2].value());
    }
    // Boss carried through; interior pore still hollow.
    assert!((top - 3.5).abs() < 0.2, "boss tip at z = {top}");
    assert!(nearest > 1.0 - 0.25, "a node intruded the pore at r = {nearest}");
}
