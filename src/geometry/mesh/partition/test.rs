use crate::{
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Bisection, Connectivity, Mesh, Partition},
    },
    math::Tensor,
};
use std::f64::consts::FRAC_PI_6;

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
fn rcb_recovers_octants() {
    let mesh = blocks([8; 3]);
    let partition = mesh.partition_rcb(8);
    let quality = partition.quality(&mesh);
    assert_eq!(quality.sizes, vec![64; 8]);
    let boxed = mesh.partition_box([2; 3]);
    assert_eq!(
        quality.interface_nodes,
        boxed.quality(&mesh).interface_nodes
    );
    let same = |a: usize, b: usize| {
        (partition.part_of(a) == partition.part_of(b)) == (boxed.part_of(a) == boxed.part_of(b))
    };
    (0..512).for_each(|element| assert!(same(0, element) && same(511, element)));
}

#[test]
fn rcb_is_balanced_for_any_part_count() {
    let mesh = blocks([6, 5, 4]);
    let elements = mesh.number_of_elements();
    (1..=elements.min(48)).for_each(|count| {
        let quality = mesh.partition_rcb(count).quality(&mesh);
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
    assert_eq!(mesh.partition_rcb(1).elements_parts(), vec![0; 12]);
    let mut parts = mesh.partition_rcb(12).elements_parts().to_vec();
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
            mesh.partition_rcb(count).quality(&mesh).disconnected_parts,
            0
        )
    });
}

#[test]
fn rib_beats_rcb_on_rotated_slab() {
    let mesh = rotated(&blocks([32, 8, 1]), FRAC_PI_6);
    let coordinate = mesh
        .partition_bisection(8, Bisection::Coordinate)
        .quality(&mesh);
    let principal = mesh
        .partition_bisection(8, Bisection::Principal)
        .quality(&mesh);
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
    let partition = mesh.partition_rcb(2);
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
    let partition = mesh.partition_rcb(5);
    (0..5).for_each(|part| {
        partition
            .part_elements(part)
            .iter()
            .for_each(|&element| assert_eq!(partition.part_of(element), part))
    });
    assert_eq!(
        (0..5)
            .map(|part| partition.part_elements(part).len())
            .sum::<usize>(),
        12
    );
}

#[test]
fn part_mesh_extracts_subdomain_with_node_map() {
    let mesh = blocks([4, 2, 2]);
    let partition = mesh.partition_rcb(2);
    (0..2).for_each(|part| {
        let (submesh, old_nodes) = partition.part_mesh(&mesh, part);
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
