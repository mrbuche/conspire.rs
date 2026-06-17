use crate::geometry::{
    grid::Voxels,
    mesh::{Tessellation, differential::laplace::Weighting, smooth::Smoothing},
};
use std::collections::HashMap;

fn distinct_positions(tessellation: &Tessellation) -> usize {
    tessellation
        .mesh()
        .coordinates()
        .into_iter()
        .map(|point| [point[0].to_bits(), point[1].to_bits(), point[2].to_bits()])
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn taubin() -> Smoothing {
    Smoothing::Taubin {
        iterations: 4,
        pass_band: 0.1,
        scale: 0.5,
        weighting: Weighting::Uniform,
    }
}

fn is_closed_manifold(tessellation: &Tessellation) -> bool {
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    for triangle in tessellation.mesh().connectivities()[0].iter() {
        for edge in 0..3 {
            let (a, b) = (triangle[edge], triangle[(edge + 1) % 3]);
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    edges.values().all(|&count| count == 2)
}

#[test]
fn single_voxel_is_a_cube() {
    let tessellation = Tessellation::from(Voxels::new(vec![1], [1, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 8);
    assert_eq!(tessellation.mesh().number_of_elements(), 12);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn void_neighbor_is_faceted_but_void_is_not() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 0], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 8);
    assert_eq!(tessellation.mesh().number_of_elements(), 12);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn same_region_welds_into_one_shell() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 1], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 12);
    assert_eq!(tessellation.mesh().number_of_elements(), 20);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn material_wall_is_doubled_into_two_closed_shells() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 2], [2, 1, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 16);
    assert_eq!(tessellation.mesh().number_of_elements(), 24);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn checkerboard_edge_splits_into_separate_cubes() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 0, 0, 1], [2, 2, 1]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 16);
    assert_eq!(tessellation.mesh().number_of_elements(), 24);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn corner_touch_vertex_splits_into_separate_cubes() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 0, 0, 0, 0, 0, 0, 1], [2, 2, 2]));
    assert_eq!(tessellation.mesh().number_of_nodes(), 16);
    assert_eq!(tessellation.mesh().number_of_elements(), 24);
    assert!(is_closed_manifold(&tessellation));
}

#[test]
fn smoothing_preserves_topology_and_moves_vertices() {
    let mut tessellation = Tessellation::from(Voxels::new(vec![1; 27], [3, 3, 3]));
    let nodes = tessellation.mesh().number_of_nodes();
    let elements = tessellation.mesh().number_of_elements();
    let before = tessellation.mesh().coordinates().clone();
    tessellation.smooth(taubin());
    assert_eq!(tessellation.mesh().number_of_nodes(), nodes);
    assert_eq!(tessellation.mesh().number_of_elements(), elements);
    assert!(is_closed_manifold(&tessellation));
    assert!(
        (&before)
            .into_iter()
            .zip(tessellation.mesh().coordinates())
            .any(|(a, b)| a != b)
    );
}

#[test]
fn welded_smoothing_keeps_coincident_copies_together() {
    let mut welded = Tessellation::from(Voxels::new(vec![1, 2], [2, 1, 1]));
    let mut split = Tessellation::from(Voxels::new(vec![1, 2], [2, 1, 1]));
    let coincident = distinct_positions(&welded);
    assert_eq!(welded.mesh().number_of_nodes(), 16);
    assert_eq!(coincident, 12);
    welded.smooth_welded(taubin());
    split.smooth(taubin());
    assert_eq!(distinct_positions(&welded), coincident);
    assert!(distinct_positions(&split) > coincident);
}

#[test]
fn l_shaped_region_stays_one_closed_manifold() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 1, 1, 0], [2, 2, 1]));
    assert_eq!(tessellation.mesh().number_of_elements(), 28);
    assert!(is_closed_manifold(&tessellation));
}
