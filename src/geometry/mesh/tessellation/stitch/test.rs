use crate::geometry::{
    mesh::tessellation::from::test::tessellation,
    ntree::{Balancing, CurvatureSizing},
};

#[test]
fn fitted_surface_produces_a_core_and_a_matching_trimesh() {
    let tessellation = tessellation();
    let (core, surface) = tessellation
        .fitted_surface(Balancing::Strong, 4.0, CurvatureSizing::default(), 3)
        .unwrap();
    assert!(core.number_of_elements() > 0);
    assert!(surface.number_of_elements() > 0);
}
