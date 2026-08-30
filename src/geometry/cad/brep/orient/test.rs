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
