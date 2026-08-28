use crate::{
    geometry::{
        Coordinate, Direction,
        csg::{
            Cuboid, Cylinder, Primitive, Sphere,
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

/// A 4-cube with a cylindrical boss on its +z face.
fn block_with_boss() -> Union<Primitive> {
    Union::new(vec![
        Cuboid::new(point([-2.0; 3]), point([2.0; 3])).unwrap().into(),
        Cylinder::new(point([0.0, 0.0, 2.0]), axis([0.0, 0.0, 1.0]), 1.0, 1.5)
            .unwrap()
            .into(),
    ])
    .unwrap()
}

#[test]
fn heterogeneous_union_signed_distance() {
    let oracle = block_with_boss().oracle().unwrap();
    // Inside the cube.
    assert!(oracle.signed_distance(&point([0.0; 3])) > 0.0);
    // Inside the boss, above the cube's top face.
    assert!(oracle.signed_distance(&point([0.0, 0.0, 3.0])) > 0.0);
    // Beside the boss, above the cube: outside both.
    assert!(oracle.signed_distance(&point([1.8, 0.0, 3.0])) < 0.0);
}

#[test]
fn block_with_boss_and_pore_meshes() {
    let solid = Difference::new(block_with_boss(), Sphere::new(point([0.0; 3]), 1.0).unwrap());
    let mesh = solid
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

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    let mut nearest = f64::INFINITY;
    for coordinate in mesh.coordinates() {
        let radius = (0..3)
            .map(|k| coordinate[k].value().powi(2))
            .sum::<f64>()
            .sqrt();
        nearest = nearest.min(radius);
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }

    // The cylindrical boss carried through the union: the mesh reaches z = 3.5.
    assert!((high[2] - 3.5).abs() < 0.2, "boss tip at z = {}", high[2]);
    // The cube walls are intact on the other axes.
    assert!((high[0] - 2.0).abs() < 0.15 && (low[0] + 2.0).abs() < 0.15);
    // The spherical pore is hollow.
    assert!(nearest > 1.0 - 0.25, "a node intruded the pore at r = {nearest}");
}
