use super::conflicts;

#[test]
fn four_cell_overlap() {
    assert!(conflicts([0, 1, 1]));
}

#[test]
fn two_cell_overlap() {
    assert!(conflicts([0, 0, 1]));
}

#[test]
fn one_cell_overlap() {
    assert!(conflicts([1, 1, 1]));
}

#[test]
fn whole_face_tangent() {
    assert!(!conflicts([2, 0, 0]));
}

#[test]
fn half_face_tangent() {
    assert!(conflicts([2, 1, 0]));
}

#[test]
fn quarter_face_tangent() {
    assert!(conflicts([2, 1, 1]));
}

#[test]
fn whole_edge_tangent() {
    assert!(!conflicts([2, 2, 0]));
}

#[test]
fn half_edge_tangent() {
    assert!(conflicts([2, 2, 1]));
}

#[test]
fn corner_tangent() {
    assert!(!conflicts([2, 2, 2]));
}

#[test]
fn disjoint() {
    assert!(!conflicts([3, 0, 0]));
    assert!(!conflicts([0, 4, 0]));
}

#[test]
fn quadtree_edge_overlap() {
    assert!(conflicts([0, 1]));
}

#[test]
fn quadtree_one_cell_overlap() {
    assert!(conflicts([1, 1]));
}

#[test]
fn quadtree_whole_edge_tangent() {
    assert!(!conflicts([2, 0]));
}

#[test]
fn quadtree_half_edge_tangent() {
    assert!(conflicts([2, 1]));
}

#[test]
fn quadtree_corner_tangent() {
    assert!(!conflicts([2, 2]));
}

#[test]
fn binary_overlap() {
    assert!(conflicts([1]));
}

#[test]
fn binary_tangent() {
    assert!(!conflicts([2]));
}

#[test]
fn binary_disjoint() {
    assert!(!conflicts([3]));
}
