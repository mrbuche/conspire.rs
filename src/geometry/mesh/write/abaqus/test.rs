use crate::{
    geometry::mesh::{Connectivity, Mesh, Output},
    io::Write,
};
use std::fs::read_to_string;

fn mixed() -> Mesh<3> {
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
    Mesh::from((connectivities, coordinates))
}

#[test]
fn mixed_input_deck() {
    let path = "target/mixed.inp";
    mixed().write(Output::Abaqus(path)).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("*Node\n"));
    assert!(contents.contains("1, 0, 0, 0\n"));
    assert!(contents.contains("11, 2, 0.5, 0.5\n"));
    assert!(contents.contains("*Element, type=C3D8, elset=BLOCK1\n1, 1, 2, 3, 4, 5, 6, 7, 8\n"));
    assert!(contents.contains("*Element, type=C3D6, elset=BLOCK2\n2, 5, 6, 9, 8, 7, 10\n"));
    assert!(contents.contains("*Element, type=C3D5, elset=BLOCK3\n3, 2, 3, 7, 6, 11\n"));
    assert!(contents.contains("*Element, type=C3D4, elset=BLOCK4\n4, 2, 3, 11, 12\n"));
}
