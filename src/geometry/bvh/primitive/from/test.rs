use crate::geometry::{bvh::primitive::Primitives, mesh::test::mesh};

#[test]
fn mesh_ref() {
    let _ = Primitives::from(&mesh());
}
