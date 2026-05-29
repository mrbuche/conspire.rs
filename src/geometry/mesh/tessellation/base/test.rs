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
    // Tessellation has one block matching mesh's one Triangular block;
    // its normals are the per-element list.
    normals[0]
        .iter()
        .zip(NORMALS.iter())
        .for_each(|(a, b)| assert_eq!(a, b))
}
