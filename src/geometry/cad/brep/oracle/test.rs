use crate::{
    geometry::{
        Coordinate,
        cad::brep::test::{capped_cylinder, cone, partial_cylinder, unit_cube},
        solid::SolidOracle,
    },
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
    let (point, normal) = oracle
        .project(&Coordinate::from([0.25, 0.75, 0.0]))
        .unwrap();
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

#[test]
fn signs_a_capped_cylinder_from_the_nearest_face() {
    let oracle = capped_cylinder(2.0, 5.0).oracle().unwrap();
    // Positive inside, magnitude the distance to the nearer wall/cap.
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, 2.5])) - 2.0).abs() < 1e-9);
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, 0.5])) - 0.5).abs() < 1e-9);
    // Negative outside.
    assert!((oracle.signed_distance(&Coordinate::from([5.0, 0.0, 2.5])) + 3.0).abs() < 1e-9);
    assert!((oracle.signed_distance(&Coordinate::from([0.0, 0.0, -1.0])) + 1.0).abs() < 1e-9);
}

#[test]
fn projects_onto_a_cylindrical_wall() {
    let oracle = capped_cylinder(2.0, 5.0).oracle().unwrap();
    let (point, normal) = oracle.project(&Coordinate::from([3.0, 0.0, 2.0])).unwrap();
    assert!(close(&components(&point), &[2.0, 0.0, 2.0]));
    assert!(close(&components(&normal), &[1.0, 0.0, 0.0]));
}

#[test]
fn accepts_a_full_cone_face() {
    assert!(cone(3.0, 1.0, 4.0).oracle().is_ok());
}

#[test]
fn accepts_a_partial_cylindrical_face() {
    assert!(partial_cylinder(2.0, 5.0, std::f64::consts::FRAC_PI_2).oracle().is_ok());
}

#[test]
fn a_partial_cylindrical_face_has_no_phantom_wall_in_the_gap() {
    let angle = std::f64::consts::FRAC_PI_2;
    let oracle = partial_cylinder(2.0, 5.0, angle).oracle().unwrap();
    // Just past the trimmed edge, still well inside the untrimmed gap: a
    // phantom full wall would report this point as already on the surface.
    let query_angle = angle + 0.5;
    let query = Coordinate::from([2.0 * query_angle.cos(), 2.0 * query_angle.sin(), 2.5]);
    let (point, _) = oracle.project(&query).unwrap();
    let [x, y, _] = components(&point);
    assert!((y.atan2(x) - angle).abs() < 1e-6);
}

#[test]
fn axial_span_rejects_a_face_with_no_vertices() {
    assert!(super::axial_span(&[], [0.0; 3], [0.0, 0.0, 1.0]).is_err());
}

#[test]
fn axial_span_rejects_a_degenerate_zero_height_face() {
    let points: Vec<[f64; 3]> = vec![[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0], [0.0, 1.0, 3.0]];
    assert!(super::axial_span(&points, [0.0; 3], [0.0, 0.0, 1.0]).is_err());
}
