use super::ReadVtkMultiBlock;
use crate::geometry::mesh::{
    Connectivity, Mesh, Output, Vtk, write::vtk::multi_block::WriteVtkMultiBlock,
};

#[test]
fn round_trip_side_sets() {
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into();
    let mut mesh = Mesh::from((connectivities, coordinates));
    let side_sets = vec![vec![(0, 4), (0, 5)], vec![(0, 0), (0, 1), (0, 2), (0, 3)]];
    mesh.set_side_sets(side_sets.clone().into());
    let path = "target/read_multi_block_round_trip.vtm";
    mesh.write_vtk_multi_block(path).unwrap();
    let read = Mesh::<3>::read_vtk_multi_block(path).unwrap();
    assert_eq!(read.number_of_nodes(), 8);
    assert_eq!(read.side_sets(), side_sets.as_slice());
    assert_eq!(read.side_set_numbers(), Some([1, 2].as_slice()));
}

#[test]
fn round_trip_side_set_numbers() {
    use crate::geometry::mesh::SideSets;
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into();
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.set_side_sets(SideSets::from((
        vec![vec![(0, 4), (0, 5)], vec![(0, 0)]],
        vec![10, 20],
    )));
    let path = "target/read_multi_block_side_set_numbers.vtm";
    mesh.write_vtk_multi_block(path).unwrap();
    let read = Mesh::<3>::read_vtk_multi_block(path).unwrap();
    assert_eq!(read.side_set_numbers(), Some([10, 20].as_slice()));
}

#[test]
fn round_trip_no_side_sets() {
    let path = "target/read_multi_block_no_side_sets.vtm";
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    let mesh = Mesh::from((connectivities, coordinates));
    mesh.write_vtk_multi_block(path).unwrap();
    let read = Mesh::<3>::read_vtk_multi_block(path).unwrap();
    assert_eq!(read.number_of_nodes(), 3);
    assert!(read.side_sets().is_empty());
}

#[test]
fn round_trip_polyhedral_volume_block() {
    let elements_faces = vec![vec![0_usize, 1, 2, 3, 4, 5]];
    let faces_nodes = vec![
        vec![0_usize, 1, 2, 3],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    let connectivities = vec![Connectivity::Polyhedral(
        (elements_faces.clone(), faces_nodes).into(),
    )];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into();
    let mesh = Mesh::from((connectivities, coordinates));
    let path = "target/read_multi_block_polyhedral.vtm";
    mesh.write_vtk_multi_block(path).unwrap();
    let read = Mesh::<3>::read_vtk_multi_block(path).unwrap();
    assert_eq!(read.number_of_nodes(), 8);
    match &read.connectivities()[0] {
        Connectivity::Polyhedral(poly) => assert!(poly.iter().eq(elements_faces.iter())),
        _ => panic!("expected Polyhedral block"),
    }
}

#[test]
fn read_via_input_enum() {
    use crate::io::Write;
    let path = "target/read_multi_block_input_enum.vtm";
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    let mesh = Mesh::from((connectivities, coordinates));
    mesh.write(Output::Vtk(Vtk::MultiBlock(path))).unwrap();
    let read = Mesh::<3>::try_from(crate::geometry::mesh::Input::VtkMultiBlock(path)).unwrap();
    assert_eq!(read.number_of_nodes(), 3);
}
