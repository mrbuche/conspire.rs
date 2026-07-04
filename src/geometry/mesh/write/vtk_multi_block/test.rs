use super::WriteVtkMultiBlock;
use crate::geometry::mesh::{Connectivity, Mesh};
use std::fs::read_to_string;

#[test]
fn writes_volume_and_side_set_blocks() {
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
    mesh.set_side_sets(vec![vec![(0, 4), (0, 5)]].into());
    mesh.write_vtk_multi_block("target/multi_block.vtm")
        .unwrap();
    let index = read_to_string("target/multi_block.vtm").unwrap();
    assert!(index.contains("vtkMultiBlockDataSet"));
    assert!(index.contains("file=\"multi_block.vtu\""));
    assert!(index.contains("file=\"multi_block_side_set_1.vtp\""));
    assert!(read_to_string("target/multi_block.vtu").is_ok());
    let side_set = read_to_string("target/multi_block_side_set_1.vtp").unwrap();
    assert!(side_set.contains("PolyData"));
    assert!(side_set.contains("NumberOfPolys=\"2\""));
}

#[test]
fn side_set_of_edges_uses_lines() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into();
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.set_side_sets(vec![vec![(0, 0)]].into());
    mesh.write_vtk_multi_block("target/multi_block_edges.vtm")
        .unwrap();
    let side_set = read_to_string("target/multi_block_edges_side_set_1.vtp").unwrap();
    assert!(side_set.contains("NumberOfLines=\"1\""));
    assert!(side_set.contains("NumberOfPolys=\"0\""));
}
