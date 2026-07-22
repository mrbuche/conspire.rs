use super::{
    Surface, collapse_short_edges, edge, edge_lengths, flip_edges, split_long_edges,
    tangential_smooth,
};
use crate::math::assert::Assert;
use crate::{
    geometry::Coordinates,
    math::{Tensor, assert::AssertionError},
};
use std::collections::HashMap;

fn right_triangle(leg: f64) -> Coordinates<3> {
    Coordinates::from(vec![[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]])
}

#[test]
fn splits_only_long_edges() -> Result<(), AssertionError> {
    let mut connectivity = vec![[0, 1, 2]];
    let mut coordinates = right_triangle(3.0);
    let lengths = edge_lengths(&connectivity, &coordinates);
    let mut sizing = vec![3.0; coordinates.len()];
    split_long_edges(
        &mut connectivity,
        &mut coordinates,
        &lengths,
        &mut sizing,
        None,
    );
    assert_eq!(coordinates.len(), 4);
    Assert::default().eq_within_tols(&coordinates[3], &[1.5, 1.5, 0.0].into())?;
    assert_eq!(connectivity, vec![[1, 3, 0], [3, 2, 0]]);
    Ok(())
}

#[test]
fn leaves_short_edges_alone() {
    let mut connectivity = vec![[0, 1, 2]];
    let mut coordinates = right_triangle(3.0);
    let lengths = edge_lengths(&connectivity, &coordinates);
    let mut sizing = vec![3.75; coordinates.len()];
    split_long_edges(
        &mut connectivity,
        &mut coordinates,
        &lengths,
        &mut sizing,
        None,
    );
    assert_eq!(coordinates.len(), 3);
    assert_eq!(connectivity, vec![[0, 1, 2]]);
}

#[test]
fn three_split_makes_four_faces() {
    let mut connectivity = vec![[0, 1, 2]];
    let mut coordinates = right_triangle(4.0);
    let lengths = edge_lengths(&connectivity, &coordinates);
    let mut sizing = vec![0.75; coordinates.len()];
    split_long_edges(
        &mut connectivity,
        &mut coordinates,
        &lengths,
        &mut sizing,
        None,
    );
    assert_eq!(connectivity.len(), 4);
    assert_eq!(coordinates.len(), 6);
}

#[test]
fn flip_reduces_overvalent_hub() {
    let mut connectivity: Vec<[usize; 3]> = (1..8)
        .map(|i| [0, i, i + 1])
        .chain(std::iter::once([0, 8, 1]))
        .collect();
    let mut points = vec![[0.0, 0.0, 0.0]];
    points.extend((0..8).map(|i| {
        let angle = i as f64 * std::f64::consts::FRAC_PI_4;
        [angle.cos(), angle.sin(), 0.0]
    }));
    let coordinates = Coordinates::from(points);
    flip_edges(&mut connectivity, &coordinates, None);
    assert_eq!(connectivity.len(), 8, "flips must preserve the face count");

    let mut hub_neighbors = std::collections::HashSet::new();
    let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in &connectivity {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *edge_uses.entry(edge(u, v)).or_default() += 1;
            if u == 0 {
                hub_neighbors.insert(v);
            } else if v == 0 {
                hub_neighbors.insert(u);
            }
        }
    }
    assert_eq!(
        hub_neighbors.len(),
        7,
        "hub valence should drop from 8 to 7"
    );
    assert!(
        edge_uses.values().all(|&count| count <= 2),
        "mesh must stay manifold (each edge in at most two faces)"
    );
}

#[test]
fn collapse_merges_short_edge() -> Result<(), AssertionError> {
    let mut connectivity = vec![
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
        [5, 1, 0],
        [5, 2, 1],
        [5, 3, 2],
        [5, 0, 3],
    ];
    let mut coordinates = Coordinates::from(vec![
        [1.0, 0.0, 0.0],
        [1.1, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]);
    let lengths = edge_lengths(&connectivity, &coordinates);
    let mut sizing = vec![1.6; coordinates.len()];
    collapse_short_edges(
        &mut connectivity,
        &mut coordinates,
        &lengths,
        &mut sizing,
        None,
    );
    assert_eq!(
        connectivity.len(),
        6,
        "two incident faces should be dropped"
    );
    assert_eq!(coordinates.len(), 5, "the merged-out vertex should be gone");
    Assert::default().eq_within_tols(&coordinates[0], &[1.05, 0.1, 0.0].into())?;
    let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in &connectivity {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *edge_uses.entry(edge(u, v)).or_default() += 1;
        }
    }
    assert!(
        edge_uses.values().all(|&count| count <= 2),
        "mesh must stay manifold after collapse"
    );
    Ok(())
}

#[test]
fn smooth_relaxes_hub_to_ring_centroid() -> Result<(), AssertionError> {
    let s = 3.0_f64.sqrt() / 2.0;
    let connectivity = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 6, 1],
    ];
    let mut coordinates = Coordinates::from(vec![
        [0.5, 0.3, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, s, 0.0],
        [-0.5, s, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -s, 0.0],
        [0.5, -s, 0.0],
    ]);
    tangential_smooth(&connectivity, &mut coordinates, None);
    Assert::default().eq_within_tols(&coordinates[0], &[0.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[1], &[1.0, 0.0, 0.0].into())
}

#[test]
fn reproject_snaps_vertex_onto_surface() -> Result<(), AssertionError> {
    let surface = Surface::new(
        &[[0, 1, 2], [0, 2, 3]],
        &Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
    );
    let mut coordinates = Coordinates::from(vec![[0.3, 0.3, 0.5]]);
    surface.reproject(&mut coordinates);
    Assert::default().eq_within_tols(&coordinates[0], &[0.3, 0.3, 0.0].into())
}
