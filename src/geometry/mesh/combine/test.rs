use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Connectivity, Mesh, NodeSets, SideSets},
};

fn hexahedron(origin: [f64; 3]) -> Mesh<3> {
    let [x, y, z] = origin;
    Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        Coordinates::from(vec![
            [x, y, z],
            [x + 1.0, y, z],
            [x + 1.0, y + 1.0, z],
            [x, y + 1.0, z],
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
            [x, y + 1.0, z + 1.0],
        ]),
    ))
}

fn hexahedron_numbered(origin: [f64; 3], number: usize) -> Mesh<3> {
    let [x, y, z] = origin;
    let connectivities = Connectivities::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        vec![number],
    ));
    let coordinates = Coordinates::from(vec![
        [x, y, z],
        [x + 1.0, y, z],
        [x + 1.0, y + 1.0, z],
        [x, y + 1.0, z],
        [x, y, z + 1.0],
        [x + 1.0, y, z + 1.0],
        [x + 1.0, y + 1.0, z + 1.0],
        [x, y + 1.0, z + 1.0],
    ]);
    Mesh::from((connectivities, coordinates.into()))
}

#[test]
fn combine_offsets_node_indices_and_concatenates_coordinates() {
    let a = hexahedron([0.0; 3]);
    let b = hexahedron([10.0; 3]);
    let combined: Mesh<3> = [a, b].into_iter().collect();
    assert_eq!(combined.number_of_nodes(), 16);
    assert_eq!(combined.number_of_elements(), 2);
    assert_eq!(combined.number_of_element_blocks(), 2);
    let blocks: Vec<&Connectivity> = combined.iter().collect();
    match blocks[1] {
        Connectivity::Hexahedral(connectivity) => {
            assert_eq!(
                *connectivity.iter().next().unwrap(),
                [8, 9, 10, 11, 12, 13, 14, 15]
            );
        }
        _ => panic!(),
    }
    let coordinates = combined.coordinates();
    assert_eq!(
        coordinates[8],
        crate::geometry::Coordinate::<3>::from([10.0, 10.0, 10.0])
    );
}

#[test]
fn combine_keeps_block_numbers_only_when_every_mesh_has_them() {
    let a = hexahedron_numbered([0.0; 3], 1);
    let b = hexahedron_numbered([10.0; 3], 2);
    let combined: Mesh<3> = [a, b].into_iter().collect();
    assert_eq!(combined.blocks(), Some([1, 2].as_slice()));

    let a = hexahedron_numbered([0.0; 3], 1);
    let b = hexahedron([10.0; 3]);
    let combined: Mesh<3> = [a, b].into_iter().collect();
    assert_eq!(combined.blocks(), None);
}

#[test]
fn combine_offsets_node_sets_and_side_sets() {
    let mut a = hexahedron([0.0; 3]);
    a.set_node_sets(NodeSets::from((vec![vec![0, 1]], vec![1])));
    a.set_side_sets(SideSets::from((vec![vec![(0, 4)]], vec![1])));
    let mut b = hexahedron([10.0; 3]);
    b.set_node_sets(NodeSets::from((vec![vec![0, 1]], vec![2])));
    b.set_side_sets(SideSets::from((vec![vec![(0, 4)]], vec![2])));
    let combined: Mesh<3> = [a, b].into_iter().collect();
    assert_eq!(combined.node_sets(), &[vec![0, 1], vec![8, 9]]);
    assert_eq!(combined.node_set_numbers(), Some([1, 2].as_slice()));
    assert_eq!(combined.side_sets(), &[vec![(0, 4)], vec![(1, 4)]]);
    assert_eq!(combined.side_set_numbers(), Some([1, 2].as_slice()));
}
