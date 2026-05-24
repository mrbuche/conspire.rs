use crate::geometry::mesh::from::test::{CONNECTIVITY, COORDINATES, mesh};

#[test]
fn connectivity_and_coordinates() {
    let (connectivity, coordinates) = mesh().into();
    assert_eq!(connectivity, CONNECTIVITY.to_vec());
    assert_eq!(coordinates, COORDINATES.into())
}
