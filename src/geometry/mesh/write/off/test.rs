use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Input, Mesh, Output},
    },
    io::Write,
    math::Tensor,
};
use std::{fs::read_to_string, io::ErrorKind};

fn square() -> Coordinates<3> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into()
}

#[test]
fn triangles() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2], [0, 2, 3]].into())];
    let path = "target/triangles.off";
    Mesh::from((connectivities, square()))
        .write(Output::Off(path))
        .unwrap();
    let contents = read_to_string(path).unwrap();
    assert!(contents.starts_with("OFF\n4 2 0\n"));
    assert!(contents.contains("1 1 0\n"));
    assert!(contents.contains("3 0 1 2\n"));
    assert!(contents.contains("3 0 2 3\n"));
}

#[test]
fn round_trip_mixed_blocks() {
    let connectivities = vec![
        Connectivity::Triangular(vec![[0, 1, 2]].into()),
        Connectivity::Quadrilateral(vec![[0, 1, 2, 3]].into()),
    ];
    let path = "target/mixed.off";
    Mesh::from((connectivities, square()))
        .write(Output::Off(path))
        .unwrap();
    let read = Mesh::<3>::try_from(Input::Off(path)).unwrap();
    assert_eq!(read.coordinates().len(), 4);
    let blocks: Vec<_> = read.iter().collect();
    assert_eq!(blocks.len(), 2);
    assert!(matches!(blocks[0], Connectivity::Triangular(_)));
    assert!(matches!(blocks[1], Connectivity::Quadrilateral(_)));
    assert_eq!(blocks[0].iter().next().unwrap(), &[0, 1, 2]);
    assert_eq!(blocks[1].iter().next().unwrap(), &[0, 1, 2, 3]);
}

#[test]
fn rejects_volumetric_blocks() {
    let connectivities = vec![Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into())];
    let error = Mesh::from((connectivities, square()))
        .write(Output::Off("target/volumetric.off"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[test]
fn rejects_two_dimensions() {
    let connectivities = vec![Connectivity::Triangular(vec![[0, 1, 2]].into())];
    let coordinates = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]].into();
    let error = Mesh::from((connectivities, coordinates))
        .write(Output::Off("target/planar.off"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}
