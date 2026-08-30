use crate::{
    geometry::{
        Coordinate, Direction,
        csg::Cylinder,
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
fn signed_distance_radial_and_axial() {
    let cylinder = Cylinder::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 2.0, 6.0).unwrap();
    let oracle = cylinder.oracle().unwrap();
    // On the axis, mid-height: nearest surface is the lateral wall, 2 away.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 3.0])) - 2.0).abs() < 1e-12);
    // Outside radially by 1.
    assert!((oracle.signed_distance(&point([3.0, 0.0, 3.0])) + 1.0).abs() < 1e-12);
    // Past the top cap by 2.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 8.0])) + 2.0).abs() < 1e-12);
    // Just inside the base cap: nearest surface is that cap.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 0.5])) - 0.5).abs() < 1e-12);
}

#[test]
fn projects_onto_wall_and_caps() {
    let cylinder = Cylinder::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 2.0, 6.0).unwrap();
    let oracle = cylinder.oracle().unwrap();

    let (surface, normal) = oracle.project(&point([1.9, 0.0, 3.0])).unwrap();
    assert!((surface[0].value() - 2.0).abs() < 1e-12 && surface[2].value() == 3.0);
    assert!((normal[0].value() - 1.0).abs() < 1e-12);

    let (surface, normal) = oracle.project(&point([0.5, 0.0, 5.8])).unwrap();
    assert!((surface[2].value() - 6.0).abs() < 1e-12);
    assert!((normal[2].value() - 1.0).abs() < 1e-12);
}

#[test]
fn tilted_axis_signed_distance() {
    let a = axis([1.0, 1.0, 0.0]);
    let cylinder = Cylinder::new(point([0.0; 3]), a, 1.0, (8.0f64).sqrt()).unwrap();
    let oracle = cylinder.oracle().unwrap();
    // Midpoint of the axis is (1,1,0); on it, nearest surface is the wall.
    assert!((oracle.signed_distance(&point([1.0, 1.0, 0.0])) - 1.0).abs() < 1e-9);
    // Straight up from that midpoint by 1 is on the lateral surface.
    assert!(oracle.signed_distance(&point([1.0, 1.0, 1.0])).abs() < 1e-9);
}

#[test]
fn meshes_the_cylinder() {
    let cylinder = Cylinder::new(point([0.0; 3]), axis([0.0, 0.0, 1.0]), 2.0, 6.0).unwrap();
    let mesh = cylinder
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

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    assert!((low[0] + 2.0).abs() < 0.25 && (high[0] - 2.0).abs() < 0.25);
    assert!((low[1] + 2.0).abs() < 0.25 && (high[1] - 2.0).abs() < 0.25);
    assert!(low[2].abs() < 0.1 && (high[2] - 6.0).abs() < 0.1);
}
