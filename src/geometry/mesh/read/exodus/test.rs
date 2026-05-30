use crate::geometry::{
    Coordinates, Write,
    mesh::{
        Connectivity, Input, Mesh, Output,
        from::test::{CONNECTIVITY, COORDINATES, mesh},
    },
};

#[test]
fn round_trip() {
    let original: Mesh<3> = mesh();
    original
        .write(Output::Exodus("target/read_exodus_round_trip.exo"))
        .unwrap();
    let read: Mesh<3> = Mesh::try_from(Input::Exodus("target/read_exodus_round_trip.exo")).unwrap();
    let expected_coords: Coordinates<3> = COORDINATES.into();
    assert_eq!(read.coordinates(), &expected_coords);
    match &read.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
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
        (elements_faces.clone(), faces_nodes.clone()).into(),
    )];
    let coordinates: Coordinates<3> = vec![
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
    let original = Mesh::from((connectivities, coordinates.clone()));
    original
        .write(Output::Exodus(
            "target/read_exodus_round_trip_polyhedral.exo",
        ))
        .unwrap();
    let read: Mesh<3> = Mesh::try_from(Input::Exodus(
        "target/read_exodus_round_trip_polyhedral.exo",
    ))
    .unwrap();
    assert_eq!(read.coordinates(), &coordinates);
    match &read.connectivities()[0] {
        Connectivity::Polyhedral(poly) => {
            assert!(poly.iter().eq(elements_faces.iter()));
        }
        _ => panic!("expected Polyhedral block"),
    }
}
