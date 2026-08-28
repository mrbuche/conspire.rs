use crate::{
    geometry::{Coordinate, cad::brep::test::unit_cube, mesh::buffer::fit::Oracle},
    math::TensorRank1,
};
use std::array::from_fn;

fn close(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn components<I, U>(tensor: &TensorRank1<3, I, U>) -> [f64; 3] {
    from_fn(|k| tensor[k].value())
}

#[test]
fn projects_an_exterior_point_onto_the_nearest_face() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([0.4, 0.6, 1.3])).unwrap();
    assert!(close(&components(&point), &[0.4, 0.6, 1.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, 1.0]));
}

#[test]
fn projects_an_interior_point_onto_the_nearest_face() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([0.4, 0.6, 0.85])).unwrap();
    assert!(close(&components(&point), &[0.4, 0.6, 1.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, 1.0]));
}

#[test]
fn a_point_on_a_face_projects_to_itself() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([0.25, 0.75, 0.0])).unwrap();
    assert!(close(&components(&point), &[0.25, 0.75, 0.0]));
    assert!(close(&components(&normal), &[0.0, 0.0, -1.0]));
}

#[test]
fn clamps_a_point_past_an_edge_to_the_trimming_loop() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, _) = oracle.project(&Coordinate::from([1.3, 0.5, 1.1])).unwrap();
    assert!(close(&components(&point), &[1.0, 0.5, 1.0]));
}

#[test]
fn clamps_a_point_past_a_corner_to_the_vertex() {
    let oracle = unit_cube().oracle().unwrap();
    let (point, _) = oracle.project(&Coordinate::from([1.4, 1.2, 1.3])).unwrap();
    assert!(close(&components(&point), &[1.0, 1.0, 1.0]));
}

#[test]
fn rejects_a_brep_with_no_faces() {
    let mut brep = unit_cube();
    brep.faces.clear();
    assert!(brep.oracle().is_err());
}
