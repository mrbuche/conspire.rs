use crate::geometry::{
    mesh::tessellation::from::test::tessellation,
    ntree::{Balancing, CurvatureSizing},
};

#[test]
fn fitted_surface_produces_a_core_and_a_matching_trimesh() {
    let tessellation = tessellation();
    let (core, surface, patches) = tessellation
        .fitted_surface(Balancing::Strong, 4.0, CurvatureSizing::default(), 3)
        .unwrap();
    assert!(core.number_of_elements() > 0);
    assert!(surface.number_of_elements() > 0);
    let number_of_quads = core.exterior_faces().len();
    assert_eq!(patches.quad_root.len(), number_of_quads);
    assert_eq!(patches.triangles.len(), number_of_quads);

    // Every root quad has a non-empty patch; merged quads have none.
    (0..number_of_quads).for_each(|quad| {
        let root = patches.quad_root[quad];
        assert_eq!(patches.quad_root[root], root, "root is not idempotent");
        if quad == root {
            assert!(
                !patches.triangles[quad].is_empty(),
                "root quad {quad} has no assigned triangles"
            );
        } else {
            assert!(
                patches.triangles[quad].is_empty(),
                "merged quad {quad} still holds triangles"
            );
        }
    });

    // Every surface triangle is assigned to exactly one root quad's patch.
    let number_of_triangles = surface.number_of_elements();
    let mut seen = vec![false; number_of_triangles];
    patches.triangles.iter().flatten().for_each(|&triangle| {
        assert!(!seen[triangle], "triangle {triangle} assigned twice");
        seen[triangle] = true;
    });
    assert!(seen.into_iter().all(|assigned| assigned));
}
