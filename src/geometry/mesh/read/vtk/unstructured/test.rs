use super::ReadVtkUnstructured;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::write;

fn first_element(mesh: &Mesh<3>, block: usize) -> &[usize] {
    mesh.iter().nth(block).unwrap().iter().next().unwrap()
}

#[test]
fn round_trip_mixed() {
    let connectivities = vec![
        Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        Connectivity::Wedge(vec![[4, 5, 8, 7, 6, 9]].into()),
        Connectivity::Pyramidal(vec![[1, 2, 6, 5, 10]].into()),
        Connectivity::Tetrahedral(vec![[1, 2, 10, 11]].into()),
    ];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 2.0],
        [0.5, 1.0, 2.0],
        [2.0, 0.5, 0.5],
        [1.5, 0.5, -1.0],
    ]
    .into();
    let path = "target/round_trip.vtu";
    Mesh::from((connectivities, coordinates))
        .write(Output::VtkUnstructured(path))
        .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 12);
    assert_eq!(mesh.number_of_element_blocks(), 4);
    assert_eq!(mesh.number_of_elements(), 4);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(first_element(&mesh, 1), [4, 5, 8, 7, 6, 9]);
    assert_eq!(first_element(&mesh, 2), [1, 2, 6, 5, 10]);
    assert_eq!(first_element(&mesh, 3), [1, 2, 10, 11]);
    let coordinates = mesh.coordinates();
    assert_eq!(
        [coordinates[10][0], coordinates[10][1], coordinates[10][2]],
        [2.0, 0.5, 0.5]
    );
}

#[test]
fn reads_ascii() {
    let path = "target/ascii.vtu";
    write(
        path,
        "<?xml version=\"1.0\"?>\n\
         <VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\">\n\
         <UnstructuredGrid><Piece NumberOfPoints=\"4\" NumberOfCells=\"1\">\n\
         <Points>\n\
         <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\
         0 0 0 1 0 0 0 1 0 0 0 1</DataArray>\n\
         </Points>\n\
         <Cells>\n\
         <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">0 1 2 3</DataArray>\n\
         <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">4</DataArray>\n\
         <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">10</DataArray>\n\
         </Cells></Piece></UnstructuredGrid></VTKFile>\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3]);
}

#[test]
fn compressed_is_unsupported() {
    let path = "target/compressed.vtu";
    write(
        path,
        "<VTKFile type=\"UnstructuredGrid\" compressor=\"vtkZLibDataCompressor\"></VTKFile>",
    )
    .unwrap();
    assert!(Mesh::<3>::read_vtk_unstructured(path).is_err());
}

#[test]
fn round_trip_node_sets() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 2, 3]].into())];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    .into();
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    let path = "target/round_trip_node_sets.vtu";
    mesh.write(Output::VtkUnstructured(path)).unwrap();
    let read = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(read.node_sets(), &[vec![0, 1], vec![2, 3]]);
}

#[test]
fn reads_point_data_node_sets_ascii() {
    let path = "target/ascii_node_sets.vtu";
    write(
        path,
        "<?xml version=\"1.0\"?>\n\
         <VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\">\n\
         <UnstructuredGrid><Piece NumberOfPoints=\"4\" NumberOfCells=\"1\">\n\
         <PointData>\n\
         <DataArray type=\"UInt8\" Name=\"NodeSet1\" format=\"ascii\">1 1 0 0</DataArray>\n\
         <DataArray type=\"UInt8\" Name=\"NodeSet2\" format=\"ascii\">0 0 1 1</DataArray>\n\
         </PointData>\n\
         <Points>\n\
         <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\
         0 0 0 1 0 0 0 1 0 1 1 0</DataArray>\n\
         </Points>\n\
         <Cells>\n\
         <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">0 1 2 1 2 3</DataArray>\n\
         <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">3 6</DataArray>\n\
         <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">5 5</DataArray>\n\
         </Cells></Piece></UnstructuredGrid></VTKFile>\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.node_sets(), &[vec![0, 1], vec![2, 3]]);
}
