use crate::geometry::{
    mesh::tessellation::from::test::tessellation,
    ntree::{Balancing, CurvatureSizing},
};

#[test]
fn fitted_surface_produces_a_core_and_a_matching_trimesh() {
    let tessellation = tessellation();
    let stitch = tessellation
        .fitted_surface(Balancing::Strong, 4.0, CurvatureSizing::default(), 8)
        .unwrap();
    let (core, surface, quads, patches, walls) = (
        stitch.core,
        stitch.surface,
        stitch.quads,
        stitch.patches,
        stitch.walls,
    );
    assert!(core.number_of_elements() > 0);
    assert!(surface.number_of_elements() > 0);
    let number_of_quads = quads.len();
    assert_eq!(patches.quad_root.len(), number_of_quads);
    assert_eq!(patches.triangles.len(), number_of_quads);
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
    let number_of_triangles = surface.number_of_elements();
    let mut seen = vec![false; number_of_triangles];
    patches.triangles.iter().flatten().for_each(|&triangle| {
        assert!(!seen[triangle], "triangle {triangle} assigned twice");
        seen[triangle] = true;
    });
    assert!(seen.into_iter().all(|assigned| assigned));

    assert!(!walls.is_empty());
    let number_of_core_nodes = core.number_of_nodes();
    walls.iter().for_each(|wall| {
        let [a, b] = wall.pair;
        assert_ne!(a, b, "wall pairs a patch with itself");
        assert_eq!(patches.quad_root[a], a, "wall pair {a} is not a root");
        assert_eq!(patches.quad_root[b], b, "wall pair {b} is not a root");
        assert!(
            wall.polygon.len() >= 3,
            "degenerate wall {:?}",
            wall.polygon
        );
        let inner = wall
            .polygon
            .iter()
            .filter(|&&node| node < number_of_core_nodes)
            .count();
        assert!(
            inner >= 1 && inner < wall.polygon.len(),
            "wall {:?} does not bridge core and surface",
            wall.polygon
        );
    });
}
