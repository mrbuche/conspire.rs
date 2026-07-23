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
        .for_each(|&node| assert!((coordinates[node].norm() - 1.0).abs() < 0.01));
    let tables = tessellation.tables(&mesh, &classes, &snapped).unwrap();
    assert!(tables.crossings().is_empty());
    let result = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 0)
}
