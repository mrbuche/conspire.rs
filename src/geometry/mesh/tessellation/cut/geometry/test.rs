use super::super::test::{hexahedron, sphere};
use super::contained;

#[test]
fn containment() {
    let tessellation = sphere(3);
    let straddling = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    assert!(!contained(&straddling, &tessellation.classify(&straddling)));
    let enclosing = hexahedron([-2.0; 3], [2.0; 3]);
    assert!(!contained(&enclosing, &tessellation.classify(&enclosing)));
    let outside = hexahedron([2.0; 3], [3.0; 3]);
    assert!(contained(&outside, &tessellation.classify(&outside)))
}
