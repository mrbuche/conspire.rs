use crate::geometry::mesh::{
    // from::test::mesh as trimesh,
    tessellation::from::test::{NORMALS, tessellation},
};

#[test]
fn mesh() {
    // let mesh = trimesh();
    let tessellation = tessellation();
    let _ = tessellation.mesh();
    // assert_eq!(tessellation.mesh(), &mesh)
}

#[test]
fn normals() {
    assert_eq!(tessellation().normals(), &NORMALS.into())
}
