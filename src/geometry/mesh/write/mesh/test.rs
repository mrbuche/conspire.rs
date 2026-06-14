use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::read_to_string;

#[test]
fn triangles_round_trip() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [0, 2, 3]].into())];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into();
    let path = "target/triangles.mesh";
    Mesh::from((connectivities, coordinates))
        .write(Output::Mesh(path))
        .unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.starts_with("MeshVersionFormatted 2"));
    assert!(contents.contains("Dimension 3"));
    assert!(contents.contains("Vertices\n4\n"));
    assert!(contents.contains("Triangles\n2\n"));
    assert!(contents.contains("1 2 3 1\n"));
    assert!(contents.contains("1 3 4 1\n"));
    assert!(contents.trim_end().ends_with("End"));
}

#[test]
fn merges_blocks_of_same_type() {
    let connectivities = vec![
        Connectivity::Triangular(vec![[0, 1, 2]].into()),
        Connectivity::Triangular(vec![[0, 2, 3]].into()),
    ];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into();
    let path = "target/merged.mesh";
    Mesh::from((connectivities, coordinates))
        .write(Output::Mesh(path))
        .unwrap();
    let contents = read_to_string(path).unwrap();
    assert_eq!(contents.matches("Triangles").count(), 1);
    assert!(contents.contains("Triangles\n2\n"));
    assert!(contents.contains("1 2 3 1\n"));
    assert!(contents.contains("1 3 4 2\n"));
}

#[test]
fn mixed_hex_wedge_pyramid_tet() {
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
    let path = "target/mixed.mesh";
    Mesh::from((connectivities, coordinates))
        .write(Output::Mesh(path))
        .unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("Vertices\n12\n"));
    assert!(contents.contains("Hexahedra\n1\n1 2 3 4 5 6 7 8 1\n"));
    assert!(contents.contains("Prisms\n1\n5 6 9 8 7 10 2\n"));
    assert!(contents.contains("Pyramids\n1\n2 3 7 6 11 3\n"));
    assert!(contents.contains("Tetrahedra\n1\n2 3 11 12 4\n"));
    assert!(contents.trim_end().ends_with("End"));
}

#[test]
fn polyhedral_is_unsupported() {
    let connectivities = vec![Connectivity::Polyhedral(
        (vec![vec![0]], vec![vec![0, 1, 2]]).into(),
    )];
    let coordinates = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]].into();
    assert!(
        Mesh::from((connectivities, coordinates))
            .write(Output::Mesh("target/bad.mesh"))
            .is_err()
    );
}
