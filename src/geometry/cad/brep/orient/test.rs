use crate::geometry::cad::brep::{Brep, test::axis_aligned_box};

/// Centroid of a planar face's outer loop vertices.
fn face_centroid(brep: &Brep, face: usize) -> [f64; 3] {
    let loop_ = &brep.faces[face].bounds[0];
    let mut sum = [0.0; 3];
    for half_edge in &loop_.half_edges {
        let vertex = brep.edges[half_edge.edge].vertices[0];
        sum.iter_mut()
            .zip(0..3)
            .for_each(|(s, k)| *s += brep.vertices[vertex][k].value());
    }
    let n = loop_.half_edges.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Every face normal points away from the box centre.
fn all_outward(brep: &Brep, centre: [f64; 3]) -> bool {
    (0..brep.faces.len()).all(|face| {
        let normal = brep.faces[face].normal().unwrap();
        let centroid = face_centroid(brep, face);
        (0..3).map(|k| normal[k] * (centroid[k] - centre[k])).sum::<f64>() > 0.0
    })
}

#[test]
fn leaves_a_well_oriented_shell_alone() {
    let mut brep = axis_aligned_box([2.0, 3.0, 4.0]);
    let before: Vec<bool> = brep.faces.iter().map(|face| face.forward).collect();
    brep.orient();
    let after: Vec<bool> = brep.faces.iter().map(|face| face.forward).collect();
    assert_eq!(before, after);
    assert!(all_outward(&brep, [1.0, 1.5, 2.0]));
}

#[test]
fn flips_faces_whose_flag_is_inverted() {
    let mut brep = axis_aligned_box([2.0, 3.0, 4.0]);
    // Invert three of the six faces: the shell is now inconsistent.
    for face in [0, 2, 4] {
        brep.faces[face].forward = false;
    }
    assert!(!all_outward(&brep, [1.0, 1.5, 2.0]));
    brep.orient();
    assert!(
        all_outward(&brep, [1.0, 1.5, 2.0]),
        "orient did not restore a consistent outward normal"
    );
}

#[test]
fn flips_a_wholly_inverted_shell() {
    let mut brep = axis_aligned_box([2.0, 3.0, 4.0]);
    for face in &mut brep.faces {
        face.forward = false;
    }
    brep.orient();
    assert!(all_outward(&brep, [1.0, 1.5, 2.0]));
}

#[test]
fn face_points_covers_a_vertex_reached_only_backward() {
    use crate::geometry::{
        Coordinate, Direction,
        cad::brep::{
            Edge, Face, HalfEdge, Loop, Shell,
            curve::{Curve, Line},
            surface::{Plane, Surface},
        },
    };
    let dir = |v: [f64; 3]| Direction::const_from(v);
    let line = |a: usize, b: usize| Edge {
        vertices: [a, b],
        curve: Curve::Line(Line {
            origin: Coordinate::const_from([0.0; 3]),
            direction: dir([1.0, 0.0, 0.0]),
        }),
    };
    // Loop v0 -> v1 (e0 fwd) -> v2 (e1 bwd) -> v0 (e2 bwd): v1 is only ever an
    // edge's `vertices[1]`, so the old per-edge `vertices[0]` scan dropped it.
    let brep = Brep {
        vertices: vec![
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 1.0, 0.0]),
        ],
        edges: vec![line(0, 1), line(2, 1), line(2, 0)],
        faces: vec![Face {
            surface: Surface::Plane(Plane {
                origin: Coordinate::const_from([0.0; 3]),
                normal: dir([0.0, 0.0, 1.0]),
                reference_direction: dir([1.0, 0.0, 0.0]),
            }),
            bounds: vec![Loop {
                half_edges: vec![
                    HalfEdge { edge: 0, forward: true },
                    HalfEdge { edge: 1, forward: false },
                    HalfEdge { edge: 2, forward: false },
                ],
            }],
            poles: vec![],
            forward: true,
        }],
        shells: vec![Shell { faces: vec![0], closed: false }],
    };
    let points = super::face_points(&brep, &brep.faces[0]);
    assert!(points.contains(&[1.0, 0.0, 0.0]), "v1 was dropped");
}
