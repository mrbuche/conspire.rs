use crate::geometry::{
    grid::Voxels,
    mesh::{Mesh, NodeSets, Partition, SideSets},
};

impl Partition {
    #[cfg_attr(
        not(any(feature = "cbm", feature = "fem", feature = "vem")),
        allow(dead_code)
    )]
    pub(crate) fn from_parts_nodes(parts_nodes: Vec<Vec<usize>>) -> Self {
        let number_of_nodes = parts_nodes
            .iter()
            .flatten()
            .max()
            .map_or(0, |&node| node + 1);
        let mut nodes_parts = vec![Vec::new(); number_of_nodes];
        parts_nodes
            .iter()
            .enumerate()
            .for_each(|(part, nodes)| nodes.iter().for_each(|&node| nodes_parts[node].push(part)));
        Self {
            elements_parts: Vec::new(),
            parts_elements: vec![Vec::new(); parts_nodes.len()],
            parts_nodes,
            nodes_parts,
        }
    }
}

pub(super) fn blocks(nel: [usize; 3]) -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![1u8; nel.iter().product()], nel), None)
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
    let partition = mesh.partition_box([2; 3]);
    (0..512).for_each(|element| assert_eq!(partition.part_of(element), octant(&mesh, element, 8)));
    let quality = partition.quality(&mesh);
    assert_eq!(quality.sizes, vec![64; 8]);
    assert_eq!(quality.imbalance, 1.0);
    assert_eq!(quality.disconnected_parts, 0);
    assert_eq!(quality.interface_nodes, 3 * 81 - 3 * 9 + 1);
}

#[test]
fn box_partition_uneven_divisions() {
    let mesh = blocks([6, 4, 2]);
    let partition = mesh.partition_box([3, 2, 1]);
    assert_eq!(partition.number_of_parts(), 6);
    assert_eq!(partition.quality(&mesh).sizes, vec![8; 6]);
}

#[test]
fn quality_treats_node_sharing_elements_as_connected() {
    let mesh = blocks([2, 2, 1]);
    let partition = Partition::new(&mesh, vec![0, 1, 1, 0]);
    assert_eq!(partition.quality(&mesh).disconnected_parts, 0);
}

#[test]
fn quality_detects_disconnected_parts() {
    let mesh = blocks([4, 1, 1]);
    let bad = Partition::new(&mesh, vec![0, 1, 0, 1]).quality(&mesh);
    assert_eq!(bad.disconnected_parts, 2);
    let good = Partition::new(&mesh, vec![0, 0, 1, 1]).quality(&mesh);
    assert_eq!(good.disconnected_parts, 0);
    assert_eq!(good.interface_nodes, 4);
    assert!(bad.interface_nodes > good.interface_nodes);
}

#[test]
fn nodes_cover_elements_and_share_interfaces() {
    let mesh = blocks([4, 2, 2]);
    let partition = mesh.partition_box([2, 1, 1]);
    assert_eq!(partition.number_of_parts(), 2);
    let interface = partition.interface_nodes();
    assert_eq!(interface.len(), 9);
    interface
        .iter()
        .for_each(|&node| assert_eq!(partition.node_parts(node), [0, 1]));
    let mut all = partition.parts_nodes().concat();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), mesh.number_of_nodes());
    mesh.iter()
        .next()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(element, nodes)| {
            nodes.iter().for_each(|node| {
                assert!(
                    partition
                        .part_nodes(partition.part_of(element))
                        .contains(node)
                )
            })
        });
}

#[test]
fn part_elements_invert_the_assignment() {
    let mesh = blocks([3, 2, 2]);
    let partition = mesh.partition_box([3, 1, 1]);
    (0..3).for_each(|part| {
        partition
            .part_elements(part)
            .iter()
            .for_each(|&element| assert_eq!(partition.part_of(element), part))
    });
    assert_eq!(
        (0..3)
            .map(|part| partition.part_elements(part).len())
            .sum::<usize>(),
        12
    );
}

#[test]
fn part_extracts_subdomain_with_node_map() {
    let mesh = blocks([4, 2, 2]);
    let partition = mesh.partition_box([2, 1, 1]);
    (0..2).for_each(|part| {
        let (submesh, old_nodes) = partition.part(&mesh, part);
        assert_eq!(submesh.number_of_elements(), 8);
        assert_eq!(submesh.number_of_nodes(), 27);
        let mut sorted = old_nodes.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, partition.part_nodes(part));
        old_nodes.iter().enumerate().for_each(|(new, &old)| {
            assert_eq!(submesh.coordinates()[new], mesh.coordinates()[old])
        });
    });
}

#[test]
fn unused_part_indices_are_empty_parts() {
    let mesh = blocks([2, 1, 1]);
    let partition = Partition::new(&mesh, vec![0, 2]);
    assert_eq!(partition.number_of_parts(), 3);
    assert!(partition.part_elements(1).is_empty());
    assert!(partition.part_nodes(1).is_empty());
}

#[test]
#[should_panic(expected = "one entry per element")]
fn partition_rejects_wrong_length_assignment() {
    Partition::new(&blocks([2, 1, 1]), vec![0]);
}

fn materials() -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![5u8, 5, 7, 7], [4, 1, 1]), None)
}

#[test]
fn part_keeps_block_ids_drops_empty_blocks_and_records_global_numbers() {
    let mesh = materials();
    assert_eq!(mesh.blocks(), Some([5, 7].as_slice()));
    let partition = mesh.partition_box([2, 1, 1]);
    let (first, first_nodes) = partition.part(&mesh, 0);
    let (second, second_nodes) = partition.part(&mesh, 1);
    assert_eq!(first.blocks(), Some([5].as_slice()));
    assert_eq!(second.blocks(), Some([7].as_slice()));
    assert_eq!(
        second.iter().next().unwrap().element_numbers(),
        Some([3, 4].as_slice())
    );
    assert_eq!(first_nodes, [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]);
    assert_eq!(
        second.coordinates.numbers(),
        Some(
            second_nodes
                .iter()
                .map(|node| node + 1)
                .collect::<Vec<_>>()
                .as_slice()
        )
    );
}

#[test]
fn part_carries_node_and_side_sets() {
    let mut mesh = materials();
    mesh.set_node_sets(NodeSets::from((
        vec![vec![0, 5, 10, 15], vec![2, 7, 12, 17]],
        vec![7, 9],
    )));
    mesh.set_side_sets(SideSets::from((vec![vec![(0, 3), (3, 1)]], vec![4])));
    let partition = mesh.partition_box([2, 1, 1]);
    let (first, first_nodes) = partition.part(&mesh, 0);
    let (second, second_nodes) = partition.part(&mesh, 1);
    assert_eq!(first.node_set_numbers(), Some([7, 9].as_slice()));
    assert_eq!(second.node_set_numbers(), Some([9].as_slice()));
    first.node_sets()[0]
        .iter()
        .zip([0, 5, 10, 15])
        .for_each(|(&local, old)| assert_eq!(first_nodes[local], old));
    second.node_sets()[0]
        .iter()
        .zip([2, 7, 12, 17])
        .for_each(|(&local, old)| assert_eq!(second_nodes[local], old));
    assert_eq!(first.side_sets(), [vec![(0, 3)]]);
    assert_eq!(second.side_sets(), [vec![(1, 1)]]);
    assert_eq!(first.side_set_numbers(), Some([4].as_slice()));
}
