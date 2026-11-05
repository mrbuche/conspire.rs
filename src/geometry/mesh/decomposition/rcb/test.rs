use super::{super::test::blocks, Bisection};
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::Tensor,
};
use std::f64::consts::FRAC_PI_6;

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
