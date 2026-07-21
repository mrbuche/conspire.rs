use super::ReadVtkUnstructured;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Output, Vtk},
    io::{Write, write::Compression},
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
        .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
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
fn unknown_compressor_is_unsupported() {
    let path = "target/unknown_compressor.vtu";
    write(
        path,
        "<VTKFile type=\"UnstructuredGrid\" compressor=\"vtkLZ4DataCompressor\"></VTKFile>",
    )
    .unwrap();
    let error = match Mesh::<3>::read_vtk_unstructured(path) {
        Ok(_) => panic!("expected an unsupported-compressor error"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
}

#[test]
fn round_trip_compressed() {
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
    let path = "target/round_trip_compressed.vtu";
    Mesh::from((connectivities, coordinates))
        .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::On(path))))
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
fn round_trip_compressed_large_mesh_spans_multiple_blocks() {
    let side = 20;
    let mut connectivity = Vec::new();
    let mut coordinates = Vec::new();
    for k in 0..side {
        for j in 0..side {
            for i in 0..side {
                coordinates.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    let index = |i: usize, j: usize, k: usize| i + j * side + k * side * side;
    for k in 0..side - 1 {
        for j in 0..side - 1 {
            for i in 0..side - 1 {
                connectivity.push([
                    index(i, j, k),
                    index(i + 1, j, k),
                    index(i + 1, j + 1, k),
                    index(i, j + 1, k),
                    index(i, j, k + 1),
                    index(i + 1, j, k + 1),
                    index(i + 1, j + 1, k + 1),
                    index(i, j + 1, k + 1),
                ]);
            }
        }
    }
    let path = "target/round_trip_compressed_large.vtu";
    Mesh::<3>::from((
        vec![Connectivity::Hexahedral(connectivity.into())],
        coordinates.into(),
    ))
    .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::On(path))))
    .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), side * side * side);
    assert_eq!(
        mesh.number_of_elements(),
        (side - 1) * (side - 1) * (side - 1)
    );
    let coordinates = mesh.coordinates();
    assert_eq!(
        [
            coordinates[index(5, 6, 7)][0],
            coordinates[index(5, 6, 7)][1],
            coordinates[index(5, 6, 7)][2]
        ],
        [5.0, 6.0, 7.0]
    );
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
    mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
        .unwrap();
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

#[test]
fn round_trip_polyhedral() {
    let elements_faces = vec![vec![0_usize, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]];
    let faces_nodes = vec![
        vec![0_usize, 1, 4, 3],
        vec![6, 7, 10, 9],
        vec![0, 1, 7, 6],
        vec![1, 4, 10, 7],
        vec![4, 3, 9, 10],
        vec![3, 0, 6, 9],
        vec![1, 2, 5, 4],
        vec![7, 8, 11, 10],
        vec![1, 2, 8, 7],
        vec![2, 5, 11, 8],
        vec![5, 4, 10, 11],
        vec![4, 1, 7, 10],
    ];
    let connectivities = vec![Connectivity::Polyhedral(
        (elements_faces.clone(), faces_nodes).into(),
    )];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    .into();
    let path = "target/round_trip_polyhedral.vtu";
    Mesh::from((connectivities, coordinates))
        .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
        .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 12);
    match &mesh.connectivities()[0] {
        Connectivity::Polyhedral(poly) => assert!(poly.iter().eq(elements_faces.iter())),
        _ => panic!("expected Polyhedral block"),
    }
}

#[test]
fn round_trip_mixed_hexahedral_and_polyhedral() {
    let hex = Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into());
    let elements_faces = vec![vec![0_usize, 1, 2, 3, 4, 5]];
    let faces_nodes = vec![
        vec![8_usize, 9, 10, 11],
        vec![12, 13, 14, 15],
        vec![8, 9, 13, 12],
        vec![9, 10, 14, 13],
        vec![10, 11, 15, 14],
        vec![11, 8, 12, 15],
    ];
    let poly = Connectivity::Polyhedral((elements_faces.clone(), faces_nodes).into());
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
        [3.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    .into();
    let path = "target/round_trip_mixed_hex_polyhedral.vtu";
    Mesh::from((vec![hex, poly], coordinates))
        .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
        .unwrap();
    let mesh = Mesh::<3>::read_vtk_unstructured(path).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 2);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
    match &mesh.connectivities()[1] {
        Connectivity::Polyhedral(poly) => assert!(poly.iter().eq(elements_faces.iter())),
        _ => panic!("expected Polyhedral block"),
    }
}
