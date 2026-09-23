use super::{test_face, test_internal};
use crate::geometry::grid::marching_cubes::cell::Cell;

fn cell_with(v: [f64; 8]) -> Cell {
    let mut cell = Cell::new(2, 2);
    cell.v = v;
    cell
}

#[test]
fn a_face_with_no_diagonal_preference_follows_the_sign_of_the_face() {
    let cell = cell_with([1.0; 8]);
    assert!(test_face(&cell, 1));
    assert!(!test_face(&cell, -1));
}

#[test]
fn a_face_is_tested_against_its_own_diagonals() {
    let joined = cell_with([2.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
    assert!(test_face(&joined, 1));
    assert!(!test_face(&joined, -1));
    let separated = cell_with([1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0]);
    assert!(!test_face(&separated, 1));
    assert!(test_face(&separated, -1));
}

#[test]
fn an_interior_of_one_sign_is_connected_only_for_its_own_sign() {
    let inside = cell_with([1.0; 8]);
    assert!(test_internal(&inside, 4, 0, 0, -1));
    assert!(!test_internal(&inside, 4, 0, 0, 1));
    let outside = cell_with([-1.0; 8]);
    assert!(test_internal(&outside, 4, 0, 0, 1));
    assert!(!test_internal(&outside, 4, 0, 0, -1));
}
