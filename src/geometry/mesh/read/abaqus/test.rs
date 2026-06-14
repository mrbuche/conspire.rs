use super::ReadAbaqus;
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
    let path = "target/round_trip.inp";
    Mesh::from((connectivities, coordinates))
        .write(Output::Abaqus(path))
        .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
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
fn reads_sparse_ids_continuation_and_comments() {
    let path = "target/sparse.inp";
    write(
        path,
        "*Heading\n deck\n*Node\n\
         100, 0., 0., 0.\n200, 1., 0., 0.\n300, 1., 1., 0.\n400, 0., 1., 0.\n\
         500, 0., 0., 1.\n600, 1., 0., 1.\n700, 1., 1., 1.\n800, 0., 1., 1.\n\
         ** a comment between data\n\
         *Element, TYPE=C3D8, ELSET=Brick\n\
         5, 100, 200, 300, 400,\n   500, 600, 700, 800\n",
    )
    .unwrap();
    let mesh = Mesh::<3>::read_abaqus(path).unwrap();
    assert_eq!(mesh.number_of_nodes(), 8);
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(first_element(&mesh, 0), [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn unknown_element_type_errors() {
    let path = "target/bad.inp";
    write(path, "*Node\n1, 0., 0., 0.\n*Element, type=C3D20\n1, 1\n").unwrap();
    assert!(Mesh::<3>::read_abaqus(path).is_err());
}
