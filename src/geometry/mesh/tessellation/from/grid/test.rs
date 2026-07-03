use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{
            Connectivity, Mesh, Tessellation, differential::laplace::Weighting, smooth::Smoothing,
        },
    },
    math::TensorVec,
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
        preserve_boundary: false,
        preserve_interfaces: false,
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

fn is_manifold(tessellation: &Tessellation) -> bool {
    let mesh = tessellation.mesh();
    mesh.non_manifold_edges().is_empty()
        && mesh.non_manifold_vertices().is_empty()
        && mesh.boundary_edges().is_empty()
}

/// Exhaustively enumerate every labeling of `dims` over `nlabels` labels (0 is
/// void) and assert the faceted surface is a closed manifold in every case.
fn assert_all_configs_manifold(dims: [usize; 3], nlabels: u32) {
    let n = dims[0] * dims[1] * dims[2];
    let total = (nlabels as u64).pow(n as u32);
    for code in 0..total {
        let mut v = code;
        let data: Vec<u32> = (0..n)
            .map(|_| {
                let label = (v % nlabels as u64) as u32;
                v /= nlabels as u64;
                label
            })
            .collect();
        let tessellation = Tessellation::from(Voxels::new(data.clone(), dims));
        assert!(
            is_manifold(&tessellation),
            "non-manifold surface for dims={dims:?} data={data:?}"
        );
    }
}

#[test]
fn every_labeling_of_small_grids_is_manifold() {
    assert_all_configs_manifold([3, 2, 2], 2);
    assert_all_configs_manifold([2, 2, 2], 3);
    assert_all_configs_manifold([3, 2, 1], 3);
    assert_all_configs_manifold([3, 3, 1], 2);
}

#[test]
fn single_material_self_pinch_is_split() {
    // A single material that diagonally touches itself across a shared edge,
    // while staying connected through the bulk, previously left a non-manifold
    // edge that the merge-only weld could not separate.
    let tessellation = Tessellation::from(Voxels::new(
        vec![1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [3, 2, 2],
    ));
    assert!(is_manifold(&tessellation));
}

#[test]
fn two_materials_meeting_at_a_checkerboard_are_split() {
    // Two distinct materials meeting diagonally at a shared edge, each connected
    // through the bulk; both shells must be split at the pinch.
    let tessellation = Tessellation::from(Voxels::new(
        vec![1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        [3, 2, 2],
    ));
    assert!(is_manifold(&tessellation));
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
fn welding_tolerance_merges_near_coincident_across_buckets() {
    let mut coordinates = Coordinates::new();
    [
        [1.2601, 0.0, 0.0],
        [1.2601, 2.0, 0.0],
        [1.2601, 0.0, 2.0],
        [1.2599, 0.0, 0.0],
        [1.2599, 4.0, 0.0],
        [1.2599, 0.0, 4.0],
    ]
    .into_iter()
    .for_each(|point| coordinates.push(Coordinate::const_from(point)));
    let connectivity = Connectivity::Triangular(vec![[0, 1, 2], [3, 4, 5]].into());
    let mut tessellation = Tessellation::from(Mesh::from((vec![connectivity], coordinates)));
    assert_eq!(distinct_positions(&tessellation), 6);
    tessellation.smooth_welded_with_tolerance(
        Smoothing::Taubin {
            iterations: 0,
            pass_band: 0.1,
            scale: 0.5,
            weighting: Weighting::Uniform,
            preserve_boundary: false,
            preserve_interfaces: false,
        },
        0.01,
    );
    assert_eq!(distinct_positions(&tessellation), 5);
}

#[test]
fn l_shaped_region_stays_one_closed_manifold() {
    let tessellation = Tessellation::from(Voxels::new(vec![1, 1, 1, 0], [2, 2, 1]));
    assert_eq!(tessellation.mesh().number_of_elements(), 28);
    assert!(is_closed_manifold(&tessellation));
}
