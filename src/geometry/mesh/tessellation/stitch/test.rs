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

    assert_eq!(walls.len(), number_of_quads);
    let number_of_core_nodes = core.number_of_nodes();
    (0..number_of_quads).for_each(|quad| {
        if patches.quad_root[quad] == quad {
            assert!(!walls[quad].is_empty(), "root quad {quad} has no wall");
            walls[quad].iter().for_each(|triangle| {
                let mut nodes = triangle.to_vec();
                nodes.sort_unstable();
                nodes.dedup();
                assert_eq!(nodes.len(), 3, "degenerate wall triangle {triangle:?}");
                let inner = triangle
                    .iter()
                    .filter(|&&node| node < number_of_core_nodes)
                    .count();
                assert!(
                    inner == 1 || inner == 2,
                    "wall triangle {triangle:?} does not bridge core and surface"
                );
            });
        } else {
            assert!(
                walls[quad].is_empty(),
                "merged quad {quad} has its own wall"
            );
        }
    });
}
