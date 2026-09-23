use crate::geometry::{
    Coordinates,
    grid::Voxels,
    mesh::{Bisection, Connectivity, Mesh},
};
use crate::math::Tensor;

fn blocks(nel: [usize; 3]) -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![1u8; nel.iter().product()], nel), None)
}

fn rotated(mesh: &Mesh<3>, angle: f64) -> Mesh<3> {
    let (sin, cos) = angle.sin_cos();
    let coordinates = mesh
        .coordinates()
        .iter()
        .map(|point| {
            let (x, y) = (point[0].value(), point[1].value());
            [cos * x - sin * y, sin * x + cos * y, point[2].value()].into()
        })
        .collect::<Coordinates<3>>();
    let elements = mesh
        .iter()
        .flat_map(|block| block.iter().map(|element| element.try_into().unwrap()))
        .collect::<Vec<[usize; 8]>>();
    Mesh::from((vec![Connectivity::Hexahedral(elements.into())], coordinates))
}

fn octant(mesh: &Mesh<3>, element: usize, nel: usize) -> usize {
    let point = mesh.centroids()[element].clone();
    (0..3).fold(0, |part, axis| {
        part + (usize::from(point[axis].value() > nel as f64 / 2.0) << axis)
    })
}

#[test]
fn box_partition_matches_octants() {
    let mesh = blocks([8; 3]);
    let parts = mesh.partition_box([2; 3]);
    (0..512).for_each(|element| assert_eq!(parts[element], octant(&mesh, element, 8)));
    let quality = mesh.partition_quality(&parts);
    assert_eq!(quality.sizes, vec![64; 8]);
    assert_eq!(quality.imbalance, 1.0);
    assert_eq!(quality.disconnected_parts, 0);
    assert_eq!(quality.interface_nodes, 3 * 81 - 3 * 9 + 1);
}

#[test]
fn box_partition_uneven_divisions() {
    let mesh = blocks([6, 4, 2]);
    let parts = mesh.partition_box([3, 2, 1]);
    assert_eq!(mesh.partition_quality(&parts).sizes, vec![8; 6]);
}

#[test]
fn rcb_recovers_octants() {
    let mesh = blocks([8; 3]);
    let parts = mesh.partition_rcb(8);
    let quality = mesh.partition_quality(&parts);
    assert_eq!(quality.sizes, vec![64; 8]);
    assert_eq!(
        quality.interface_nodes,
        mesh.partition_quality(&mesh.partition_box([2; 3]))
            .interface_nodes
    );
    let box_parts = mesh.partition_box([2; 3]);
    let same = |a: usize, b: usize| (parts[a] == parts[b]) == (box_parts[a] == box_parts[b]);
    (0..512).for_each(|element| assert!(same(0, element) && same(511, element)));
}

#[test]
fn rcb_is_balanced_for_any_part_count() {
    let mesh = blocks([6, 5, 4]);
    let elements = mesh.number_of_elements();
    (1..=elements.min(48)).for_each(|count| {
        let quality = mesh.partition_quality(&mesh.partition_rcb(count));
        assert_eq!(quality.sizes.len(), count);
        assert!(
            quality
                .sizes
                .iter()
                .all(|&size| size == elements / count || size == elements.div_ceil(count)),
            "{count}: {:?}",
            quality.sizes
        );
        assert_eq!(quality.sizes.iter().sum::<usize>(), elements);
    });
}

#[test]
fn rcb_is_deterministic_on_tied_centroids() {
    let mesh = blocks([4; 3]);
    assert_eq!(mesh.partition_rcb(5), mesh.partition_rcb(5));
    assert_eq!(mesh.partition_rib(5), mesh.partition_rib(5));
}

#[test]
fn rcb_single_part_and_one_element_per_part() {
    let mesh = blocks([3, 2, 2]);
    assert_eq!(mesh.partition_rcb(1), vec![0; 12]);
    let mut parts = mesh.partition_rcb(12);
    parts.sort_unstable();
    assert_eq!(parts, (0..12).collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "parts must be between")]
fn rcb_rejects_more_parts_than_elements() {
    blocks([2, 1, 1]).partition_rcb(3);
}

#[test]
fn rcb_parts_are_connected_on_blocks() {
    let mesh = blocks([9, 4, 3]);
    (2..=12).for_each(|count| {
        assert_eq!(
            mesh.partition_quality(&mesh.partition_rcb(count))
                .disconnected_parts,
            0
        )
    });
}

#[test]
fn rib_beats_rcb_on_rotated_slab() {
    let mesh = rotated(&blocks([32, 8, 1]), std::f64::consts::FRAC_PI_6);
    let coordinate = mesh.partition_quality(&mesh.partition_bisection(8, Bisection::Coordinate));
    let principal = mesh.partition_quality(&mesh.partition_bisection(8, Bisection::Principal));
    assert_eq!(principal.sizes, vec![32; 8]);
    assert!(
        principal.interface_nodes < coordinate.interface_nodes,
        "{} vs {}",
        principal.interface_nodes,
        coordinate.interface_nodes
    );
}

#[test]
fn quality_treats_node_sharing_elements_as_connected() {
    let mesh = blocks([2, 2, 1]);
    assert_eq!(mesh.partition_quality(&[0, 1, 1, 0]).disconnected_parts, 0);
}

#[test]
fn partition_nodes_cover_elements_and_share_interfaces() {
    let mesh = blocks([4, 2, 2]);
    let parts = mesh.partition_rcb(2);
    let nodes = mesh.partition_nodes(&parts);
    assert_eq!(nodes.len(), 2);
    let shared = nodes[0]
        .iter()
        .filter(|node| nodes[1].contains(node))
        .count();
    assert_eq!(shared, 9);
    let mut all = nodes.concat();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), mesh.number_of_nodes());
    let block = mesh.iter().next().unwrap();
    block
        .iter()
        .enumerate()
        .for_each(|(element, connectivity)| {
            connectivity
                .iter()
                .for_each(|node| assert!(nodes[parts[element]].contains(node)))
        });
}

#[test]
fn quality_detects_disconnected_parts() {
    let mesh = blocks([4, 1, 1]);
    let bad = mesh.partition_quality(&[0, 1, 0, 1]);
    assert_eq!(bad.disconnected_parts, 2);
    let good = mesh.partition_quality(&[0, 0, 1, 1]);
    assert_eq!(good.disconnected_parts, 0);
    assert_eq!(good.interface_nodes, 4);
    assert!(bad.interface_nodes > good.interface_nodes);
}
