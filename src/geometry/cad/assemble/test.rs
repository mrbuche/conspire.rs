use super::assemble;
use crate::{
    geometry::{
        Coordinate,
        cad::brep::test::{axis_aligned_box, ball_at, coplanar_squares},
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

#[test]
fn a_lone_solid_is_a_body_with_no_voids() {
    let bodies = assemble(&[axis_aligned_box([2.0, 2.0, 2.0])]).unwrap();
    assert_eq!(bodies.len(), 1);
    let oracle = bodies[0].oracle().unwrap();
    assert!(oracle.signed_distance(&point([1.0, 1.0, 1.0])) > 0.0);
    assert!(oracle.signed_distance(&point([3.0, 1.0, 1.0])) < 0.0);
}

#[test]
fn interior_solids_become_voids() {
    let breps = [
        axis_aligned_box([6.0, 6.0, 6.0]),
        ball_at([2.0, 3.0, 3.0], 0.7),
        ball_at([4.0, 3.0, 3.0], 0.7),
    ];
    let bodies = assemble(&breps).unwrap();
    assert_eq!(bodies.len(), 1);
    let oracle = bodies[0].oracle().unwrap();

    // Solid material between the voids.
    assert!(oracle.signed_distance(&point([3.0, 1.0, 3.0])) > 0.0);
    // Inside each carved void.
    assert!(oracle.signed_distance(&point([2.0, 3.0, 3.0])) < 0.0);
    assert!(oracle.signed_distance(&point([4.0, 3.0, 3.0])) < 0.0);

    let mesh = bodies
        .into_iter()
        .next()
        .unwrap()
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
}

#[test]
fn an_unrecognised_solid_is_rejected() {
    assert!(assemble(&[coplanar_squares()]).is_err());
}
