use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivity, Mesh,
        from::test::{CONNECTIVITY, COORDINATES, mesh},
        read::ReadExodus,
        write::exodus::WriteExodus,
    },
};

#[test]
fn round_trip() {
    let original: Mesh<3> = mesh();
    original
        .write_exodus("target/read_exodus_round_trip.exo")
        .unwrap();
    let read: Mesh<3> = Mesh::read_exodus("target/read_exodus_round_trip.exo").unwrap();
    // Coordinates should round-trip exactly (f64 → exodus f64 → f64).
    let expected_coords: Coordinates<3> = COORDINATES.into();
    assert_eq!(read.coordinates(), &expected_coords);
    // Triangles should round-trip with same node ordering.
    match &read.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
}