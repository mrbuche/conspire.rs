use crate::{
    geometry::{Coordinate, cad::brep::test::axis_aligned_box},
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
    let area: f64 = (0..4)
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % 4];
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
fn reversed_face_flips_the_normal() {
    let brep = axis_aligned_box(EXTENTS);
    // Face 4 is the `x = 0` face; its outward normal points down -x.
    let face = brep.planar_face(&brep.faces[4]).unwrap();
    assert!(close(&components(&face.normal), &[-1.0, 0.0, 0.0]));
    assert!((face.plane_distance(&Coordinate::from([-1.0, 2.0, 4.0])) - 1.0).abs() < 1e-12);
}
