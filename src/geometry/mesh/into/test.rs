use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivity, PrimitiveConnectivity,
        from::test::{CONNECTIVITY, COORDINATES, mesh},
    },
};

#[test]
fn connectivities_and_coordinates() {
    let (connectivities, coordinates) = mesh().into();
    let expected: Coordinates<3> = COORDINATES.into();
    assert_eq!(coordinates, expected);
    match connectivities.into_iter().next().unwrap() {
        Connectivity::Triangular(PrimitiveConnectivity(t)) => {
            assert_eq!(t, CONNECTIVITY.to_vec())
        }
        _ => panic!("expected Triangular block"),
    }
}
