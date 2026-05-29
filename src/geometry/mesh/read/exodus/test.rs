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
    let read: Mesh<3> =
        Mesh::try_from(Input::Exodus("target/read_exodus_round_trip.exo")).unwrap();
    let expected_coords: Coordinates<3> = COORDINATES.into();
    assert_eq!(read.coordinates(), &expected_coords);
    match &read.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
}