use crate::geometry::{
    Coordinate,
    csg::Cuboid,
    mesh::{Connectivity, Mesh},
    solid::Partition,
};

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn cuboid(low: [f64; 3], high: [f64; 3]) -> Cuboid {
    Cuboid::new(point(low), point(high)).unwrap()
}

fn elements(mesh: &Mesh<3>) -> Vec<usize> {
    mesh.iter()
        .map(|block| match block {
            Connectivity::Hexahedral(hexahedra) => hexahedra.iter().count(),
            _ => panic!(),
        })
        .collect()
}

#[test]
fn touching_boxes_share_the_nodes_between_them() {
    let partition = Partition::new(vec![
        cuboid([0.0; 3], [1.0, 1.0, 1.0]),
        cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    ]);
    let mesh = Mesh::from_partition(
        &partition,
        &point([0.0; 3]),
        &point([2.0, 1.0, 1.0]),
        [4, 2, 2],
    )
    .unwrap();
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(elements(&mesh), [8, 8]);
    assert_eq!(mesh.number_of_nodes(), 5 * 3 * 3);
}

#[test]
fn a_later_volume_owns_the_space_it_overlaps() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [4.0; 3]), cuboid([1.0; 3], [3.0; 3])]);
    let mesh =
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(elements(&mesh), [56, 8]);
    assert_eq!(mesh.number_of_nodes(), 5 * 5 * 5);
}

#[test]
fn cells_in_no_volume_are_dropped() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]);
    let mesh =
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(elements(&mesh), [8]);
    assert_eq!(mesh.number_of_nodes(), 3 * 3 * 3);
}

#[test]
fn a_degenerate_lattice_is_refused() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.0; 3])]);
    assert!(
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([1.0; 3]), [0, 1, 1]).is_err()
    );
    assert!(
        Mesh::from_partition(
            &partition,
            &point([0.0; 3]),
            &point([1.0, 0.0, 1.0]),
            [1; 3]
        )
        .is_err()
    );
}
