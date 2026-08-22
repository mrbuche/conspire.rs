use crate::geometry::{
    mesh::test::sphere,
    ntree::{CurvatureSizing, Octree, Sizing},
};

#[test]
fn a_size_field_knows_the_depth_it_asks_for() {
    let tessellation = sphere(4, 8, 2.0);
    let ordinary = Sizing::new(&tessellation, 4.0, CurvatureSizing::default(), 0);
    assert!(ordinary.levels() <= 15);
    assert!(ordinary.fits::<u16>());
    assert!(Octree::<u16, usize>::refine(&ordinary).is_ok());
    let deep = Sizing::new(&tessellation, 1.0e6, CurvatureSizing::default(), 0);
    assert!(deep.levels() > 15);
    assert!(!deep.fits::<u16>());
    assert!(deep.fits::<u32>());
    assert_eq!(
        Octree::<u16, usize>::refine(&deep).err(),
        Some("sizing field exceeds maximum octree depth")
    );
}
