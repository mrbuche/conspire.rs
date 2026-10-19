use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Input, Mesh, NodeSet, Output, SideSet},
    },
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

fn tetrahedra() -> (Vec<Connectivity>, Coordinates<3>) {
    (
        vec![Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into())],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into(),
    )
}

#[test]
fn round_trip_node_labels() {
    let (connectivities, coordinates) = tetrahedra();
    let mut mesh = Mesh::from((
        Connectivities::from(connectivities),
        coordinates,
        vec![7, 8, 9, 10],
    ));
    mesh.set_node_sets(vec![NodeSet::new(vec![0, 2])].into());
    let path = "target/write_abaqus_node_labels.inp";
    mesh.write(Output::Abaqus(path)).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("*Node\n7, 0, 0, 0\n8, 1, 0, 0\n9, 0, 1, 0\n10, 0, 0, 1\n"));
    assert!(contents.contains("1, 7, 8, 9, 10\n"));
    assert!(contents.contains("*Nset, nset=NSET1\n7, 9\n"));
    let read = Mesh::<3>::try_from(Input::Abaqus(path)).unwrap();
    assert_eq!(read.node_labels(), Some([7, 8, 9, 10].as_slice()));
    assert_eq!(
        read.iter().next().unwrap().iter().next().unwrap(),
        [0, 1, 2, 3]
    );
    assert_eq!(read.node_sets().by_name("NSET1").unwrap().nodes(), [0, 2]);
}

#[test]
fn round_trip_element_labels() {
    let mut block_0 = Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into());
    block_0.label_elements(vec![100]);
    let mut block_1 = Connectivity::Tetrahedral(vec![[1, 2, 3, 4]].into());
    block_1.label_elements(vec![200]);
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    .into();
    let mut mesh = Mesh::from((vec![block_0, block_1], coordinates));
    mesh.set_side_sets(vec![SideSet::new(vec![(1, 2)])].into());
    let path = "target/write_abaqus_element_labels.inp";
    mesh.write(Output::Abaqus(path)).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("elset=BLOCK1\n100, 1, 2, 3, 4\n"));
    assert!(contents.contains("elset=BLOCK2\n200, 2, 3, 4, 5\n"));
    assert!(contents.contains("*Surface, type=ELEMENT, name=SURF1\n200, S"));
    let read = Mesh::<3>::try_from(Input::Abaqus(path)).unwrap();
    assert_eq!(
        read.connectivities()[0].element_labels(),
        Some([100].as_slice())
    );
    assert_eq!(
        read.connectivities()[1].element_labels(),
        Some([200].as_slice())
    );
    assert_eq!(read.side_sets().by_name("SURF1").unwrap().sides(), [(1, 2)]);
}

#[test]
fn unlabelled_input_deck() {
    let (connectivities, coordinates) = tetrahedra();
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.set_node_sets(vec![NodeSet::new(vec![0, 2])].into());
    mesh.set_side_sets(vec![SideSet::new(vec![(0, 3)])].into());
    let path = "target/write_abaqus_unlabelled.inp";
    mesh.write(Output::Abaqus(path)).unwrap();
    assert_eq!(
        read_to_string(path).unwrap(),
        "*Heading\n conspire mesh\n*Node\n1, 0, 0, 0\n2, 1, 0, 0\n3, 0, 1, 0\n4, 0, 0, 1\n\
         *Element, type=C3D4, elset=BLOCK1\n1, 1, 2, 3, 4\n\
         *Nset, nset=NSET1\n1, 3\n\
         *Surface, type=ELEMENT, name=SURF1\n1, S1\n"
    );
}

#[test]
fn partially_labelled_blocks_are_numbered() {
    let mut block_0 = Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into());
    block_0.label_elements(vec![2]);
    let block_1 = Connectivity::Tetrahedral(vec![[1, 2, 3, 4]].into());
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    .into();
    let mut mesh = Mesh::from((vec![block_0, block_1], coordinates));
    mesh.set_side_sets(vec![SideSet::new(vec![(1, 2)])].into());
    let path = "target/write_abaqus_partially_labelled.inp";
    mesh.write(Output::Abaqus(path)).unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.contains("elset=BLOCK1\n1, 1, 2, 3, 4\n"));
    assert!(contents.contains("elset=BLOCK2\n2, 2, 3, 4, 5\n"));
    let read = Mesh::<3>::try_from(Input::Abaqus(path)).unwrap();
    assert_eq!(
        read.connectivities()[0].element_labels(),
        Some([1].as_slice())
    );
    assert_eq!(
        read.connectivities()[1].element_labels(),
        Some([2].as_slice())
    );
    assert_eq!(read.side_sets().by_name("SURF1").unwrap().sides(), [(1, 2)]);
}
