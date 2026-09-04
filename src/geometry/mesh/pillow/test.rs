use crate::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh, Verdict},
};
use std::collections::HashMap;

/// A `n`-by-`n`-by-`n` grid of unit hexahedra.
fn grid(n: usize) -> Mesh<3> {
    let node = |i: usize, j: usize, k: usize| i + (n + 1) * (j + (n + 1) * k);
    let coordinates: Vec<[f64; 3]> = (0..=n)
        .flat_map(|k| {
            (0..=n).flat_map(move |j| (0..=n).map(move |i| [i as f64, j as f64, k as f64]))
        })
        .collect();
    let hexes: Vec<[usize; 8]> = (0..n)
        .flat_map(|k| {
            (0..n).flat_map(move |j| {
                (0..n).map(move |i| {
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]
                })
            })
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Hexahedral(hexes.into())],
        Coordinates::from(coordinates),
    ))
}

fn face_counts(mesh: &Mesh<3>) -> HashMap<[usize; 4], usize> {
    let mut counts = HashMap::new();
    for block in mesh.iter() {
        for element in block.iter() {
            for face in block.element_faces(element) {
                let mut key = [face[0], face[1], face[2], face[3]];
                key.sort_unstable();
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn volume(mesh: &Mesh<3>) -> f64 {
    mesh.volumes().into_iter().flatten().sum()
}

#[test]
fn pillowing_an_interior_hex_wraps_it_in_a_layer() {
    let mut mesh = grid(3);
    let before = face_counts(&mesh);
    let exterior_before = before.values().filter(|&&c| c == 1).count();
    let twins = mesh.pillow(&[13]).unwrap();
    assert_eq!(twins.len(), 8);
    assert_eq!(mesh.number_of_elements(), 27 + 6);
    assert_eq!(mesh.number_of_nodes(), 64 + 8);
    let after = face_counts(&mesh);
    assert!(after.values().all(|&c| c <= 2), "non-conforming face");
    assert_eq!(
        after.values().filter(|&&c| c == 1).count(),
        exterior_before,
        "outer surface changed"
    );
    assert!((volume(&mesh) - 27.0).abs() < 1e-9, "{}", volume(&mesh));
}

#[test]
fn pillowing_a_corner_hex_keeps_the_mesh_conforming() {
    let mut mesh = grid(3);
    let twins = mesh.pillow(&[0]).unwrap();
    assert_eq!(twins.len(), 8);
    assert_eq!(mesh.number_of_elements(), 27 + 6);
    assert!(face_counts(&mesh).values().all(|&c| c <= 2));
    assert!((volume(&mesh) - 27.0).abs() < 1e-9, "{}", volume(&mesh));
}

#[test]
fn pillowing_a_block_of_hexes_wraps_the_whole_block() {
    let mut mesh = grid(4);
    // the 2x2x2 core of a 4x4x4 grid
    let core: Vec<usize> = (1..3)
        .flat_map(|k| (1..3).flat_map(move |j| (1..3).map(move |i| i + 4 * (j + 4 * k))))
        .collect();
    let twins = mesh.pillow(&core).unwrap();
    assert_eq!(twins.len(), 26); // 3x3x3 nodes on the block's surface, minus interior
    assert_eq!(mesh.number_of_elements(), 64 + 24);
    assert!(face_counts(&mesh).values().all(|&c| c <= 2));
    assert!((volume(&mesh) - 64.0).abs() < 1e-9, "{}", volume(&mesh));
}

#[test]
fn pillow_rejects_a_non_manifold_region() {
    let mut mesh = grid(3);
    // hexes 0 (x,y,z in 0..1) and 4 (x,y in 1..2, z in 0..1) share one edge
    // but no face, so their combined boundary pinches along that edge
    assert!(mesh.pillow(&[0, 4]).is_err());
}
