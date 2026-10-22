use super::separation::{THIN, WIDE, narrowness};
use crate::geometry::{
    mesh::test::sphere,
    ntree::{CurvatureSizing, Octree, SeparationSizing, Sizing},
};

#[test]
fn a_size_field_knows_the_depth_it_asks_for() {
    let tessellation = sphere(4, 8, 2.0);
    let ordinary = Sizing::new(
        &tessellation,
        4.0,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    );
    assert!(ordinary.levels() <= 15);
    assert!(ordinary.fits::<u16>());
    assert!(Octree::<u16, usize>::refine(&ordinary).is_ok());
    let deep = Sizing::new(
        &tessellation,
        1.0e6,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    );
    assert!(deep.levels() > 15);
    assert!(!deep.fits::<u16>());
    assert!(deep.fits::<u32>());
    assert_eq!(
        Octree::<u16, usize>::refine(&deep).err(),
        Some("sizing field exceeds maximum octree depth")
    );
}

#[test]
fn creases_are_narrow_only_while_they_open_slowly() {
    assert_eq!(narrowness(0.0), 1.0);
    assert_eq!(narrowness(THIN), 1.0);
    assert_eq!(narrowness(WIDE), 0.0);
    assert_eq!(narrowness(2.0 * WIDE), 0.0);
    let steps: Vec<f64> = (0..=20)
        .map(|step| narrowness(THIN + (WIDE - THIN) * step as f64 / 20.0))
        .collect();
    assert!(steps.windows(2).all(|pair| pair[0] >= pair[1]), "{steps:?}");
    assert!(steps[10] > 0.4 && steps[10] < 0.6, "{}", steps[10]);
}
