use crate::geometry::{bvh::primitive::Primitives, mesh::from::test::mesh};

#[test]
fn mesh_ref() {
    let _ = Primitives::from(&mesh());
}
