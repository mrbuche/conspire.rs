use super::super::test::{hexahedron, sphere};
use crate::math::Tensor;

#[test]
fn snap_eliminates_sliver() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.95, -0.1, -0.1], [1.15, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let (mesh, snapped) = tessellation.snap(mesh, &classes).unwrap();
    assert_eq!(snapped.len(), 4);
    let coordinates = mesh.coordinates();
    snapped
        .iter()
        .for_each(|&node| assert!((coordinates[node].norm().value() - 1.0).abs() < 0.01));
    let tables = tessellation.tables(&mesh, &classes, &snapped).unwrap();
    assert!(tables.crossings().is_empty());
    let result = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 0)
}

#[test]
fn snap_generic_matches_snap_hexahedron() {
    let tessellation = sphere(3);
    let bounds = ([0.95, -0.1, -0.1], [1.15, 0.1, 0.1]);
    let classes = tessellation.classify(&hexahedron(bounds.0, bounds.1));
    let (expected_mesh, expected_snapped) = tessellation
        .snap(hexahedron(bounds.0, bounds.1), &classes)
        .unwrap();
    let (result_mesh, result_snapped) = tessellation
        .snap_generic(hexahedron(bounds.0, bounds.1), &classes)
        .unwrap();
    assert_eq!(result_snapped, expected_snapped);
    assert_eq!(result_mesh.coordinates(), expected_mesh.coordinates());
}
