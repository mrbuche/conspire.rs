use crate::{
    geometry::{Coordinate, cad::brep::test::{axis_aligned_box, square_with_rounded_hole}},
    math::TensorRank1,
};
use std::array::from_fn;

const EXTENTS: [f64; 3] = [2.0, 4.0, 8.0];

fn close(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
}

fn components<I, U>(tensor: &TensorRank1<3, I, U>) -> [f64; 3] {
    from_fn(|k| tensor[k].value())
}

#[test]
fn frame_rings_and_plane() {
    let brep = axis_aligned_box(EXTENTS);
    // Face 5 is the `x = 2` face, a 4x8 rectangle in the y-z plane.
    let face = brep.planar_face(&brep.faces[5]).unwrap();

    assert!(close(&components(&face.normal), &[1.0, 0.0, 0.0]));
    assert!((face.plane_distance(&Coordinate::from([3.0, 2.0, 4.0])) - 1.0).abs() < 1e-12);
    assert!(
        face.plane_distance(&Coordinate::from([2.0, 1.0, 5.0]))
            .abs()
            < 1e-12
    );

    assert_eq!(face.rings.len(), 1);
    let ring = &face.rings[0];
    assert_eq!(ring.len(), 4);
    assert!(ring.iter().all(|(_, arc)| arc.is_none()));
    let area: f64 = (0..4)
        .map(|i| {
            let a = ring[i].0;
            let b = ring[(i + 1) % 4].0;
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        .abs()
        / 2.0;
    assert!((area - 32.0).abs() < 1e-12);

    assert!(close(&components(face.aabb.minimum()), &[2.0, 0.0, 0.0]));
    assert!(close(&components(face.aabb.maximum()), &[2.0, 4.0, 8.0]));
}

#[test]
fn project_round_trips_on_the_plane() {
    let brep = axis_aligned_box(EXTENTS);
    let face = brep.planar_face(&brep.faces[5]).unwrap();
    let point = Coordinate::from([2.0, 1.5, 6.0]);
    let back = face.unproject(face.project(&point));
    assert!(close(&components(&back), &components(&point)));
}

#[test]
fn rounded_hole_trims_the_true_arc_not_a_bounding_box() {
    let brep = square_with_rounded_hole();
    let face = brep.planar_face(&brep.faces[0]).unwrap();
    assert!(!face.contains([5.0, 5.0])); // centre of the hole
    assert!(face.contains([1.0, 1.0])); // outer material
    // Outside the rounded corner's disk (radius 1 about (7, 3)): material —
    // a bounding-box hole (ignoring the arc) would wrongly call this a hole.
    assert!(face.contains([7.9, 2.5]));
    // Inside that same disk, within the arc's swept sector: still hole.
    assert!(!face.contains([7.9, 2.9]));
}

#[test]
fn nearest_boundary_snaps_to_the_arc_not_its_chord() {
    let brep = square_with_rounded_hole();
    let face = brep.planar_face(&brep.faces[0]).unwrap();
    // Straight out from the (7, 3) corner centre at -45 degrees, well past
    // the arc: the true nearest point is the radial projection onto the arc.
    // The chord between the arc's two endpoints would instead give (7.5,
    // 2.5) here — visibly different, so this discriminates the two.
    let half_diagonal = std::f64::consts::FRAC_1_SQRT_2;
    let query = [7.0 + 2.0 * half_diagonal, 3.0 - 2.0 * half_diagonal];
    let expected = [7.0 + half_diagonal, 3.0 - half_diagonal];
    let nearest = face.nearest_boundary(query);
    assert!((nearest[0] - expected[0]).abs() < 1e-9);
    assert!((nearest[1] - expected[1]).abs() < 1e-9);
}

#[test]
fn reversed_face_flips_the_normal() {
    let brep = axis_aligned_box(EXTENTS);
    // Face 4 is the `x = 0` face; its outward normal points down -x.
    let face = brep.planar_face(&brep.faces[4]).unwrap();
    assert!(close(&components(&face.normal), &[-1.0, 0.0, 0.0]));
    assert!((face.plane_distance(&Coordinate::from([-1.0, 2.0, 4.0])) - 1.0).abs() < 1e-12);
}
