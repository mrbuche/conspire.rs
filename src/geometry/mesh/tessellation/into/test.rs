use crate::geometry::mesh::{
    TriangularMesh,
    tessellation::from::test::{CONNECTIVITY, COORDINATES, NORMALS, tessellation},
};

#[test]
fn triangluar_mesh() {
    let mesh = TriangularMesh::from(tessellation());
    assert_eq!(mesh.connectivity, CONNECTIVITY.to_vec());
    assert_eq!(mesh.coordinates, COORDINATES.into())
}

#[test]
fn connectivity_and_coordinates() {
    let (connectivity, coordinates) = tessellation().into();
    assert_eq!(connectivity, CONNECTIVITY.to_vec());
    assert_eq!(coordinates, COORDINATES.into())
}

#[test]
fn connectivity_and_coordinates_and_normals() {
    let (connectivity, coordinates, normals) = tessellation().into();
    assert_eq!(connectivity, CONNECTIVITY.to_vec());
    assert_eq!(coordinates, COORDINATES.into());
    assert_eq!(normals, NORMALS.into())
}
