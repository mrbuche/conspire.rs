use super::ReadMesh;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::write;

fn first_element<'a>(mesh: &'a Mesh<3>, block: usize) -> &'a [usize] {
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
    let path = "target/round_trip.mesh";
    Mesh::from((connectivities, coordinates))
        .write(Output::Mesh(path))
        .unwrap();
    let mesh = Mesh::<3>::read_mesh(path).unwrap();
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
    assert_eq!(coordinates[11][2], -1.0);
}

#[test]
fn skips_edges_and_corners() {
    let path = "target/with_extras.mesh";
    write(
        path,
        "MeshVersionFormatted 2\nDimension 3\nVertices\n4\n\
         0 0 0 0\n1 0 0 0\n0 1 0 0\n0 0 1 0\n\
         Edges\n2\n1 2 0\n2 3 0\n\
         Corners\n1\n1\n\
         Tetrahedra\n1\n1 2 3 4 0\nEnd\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_mesh(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3]);
}

#[test]
fn dimension_mismatch_errors() {
    let path = "target/wrong_dim.mesh";
    write(
        path,
        "MeshVersionFormatted 2\nDimension 2\nVertices\n1\n0 0 0\nEnd\n",
    )
    .unwrap();
    assert!(Mesh::<3>::read_mesh(path).is_err());
}

#[test]
fn unknown_keyword_errors() {
    let path = "target/unknown.mesh";
    write(
        path,
        "MeshVersionFormatted 2\nDimension 3\nGreebles\n1\n0\nEnd\n",
    )
    .unwrap();
    assert!(Mesh::<3>::read_mesh(path).is_err());
}
