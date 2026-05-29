use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivity,
        from::test::{CONNECTIVITY, COORDINATES, mesh},
    },
};

#[test]
fn connectivities_and_coordinates() {
    let (connectivities, coordinates) = mesh().into();
    let expected: Coordinates<3> = COORDINATES.into();
    assert_eq!(coordinates, expected);
    match connectivities.into_iter().next().unwrap() {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.into_iter().eq(CONNECTIVITY))
        }
        _ => panic!("expected Triangular block"),
    }
}
