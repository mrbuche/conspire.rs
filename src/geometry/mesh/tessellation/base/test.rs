use crate::{
    geometry::mesh::tessellation::from::test::{NORMALS, tessellation},
    math::Tensor,
};

#[test]
fn mesh() {
    let tessellation = tessellation();
    let _ = tessellation.mesh();
}

#[test]
fn normals() {
    let tess = tessellation();
    let normals = tess.normals();
    normals[0]
        .iter()
        .zip(NORMALS.iter())
        .for_each(|(a, b)| assert_eq!(a, b))
}
