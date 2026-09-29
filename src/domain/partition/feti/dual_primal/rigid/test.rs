use super::{kernel, kernel_pins, removed_modes};
use crate::math::Tensor;

const CORNER: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

fn all_of(nodes: &[usize]) -> Vec<usize> {
    nodes
        .iter()
        .flat_map(|&node| (0..3).map(move |component| 3 * node + component))
        .collect()
}

#[test]
fn nothing_fixed_removes_nothing() {
    assert_eq!(removed_modes(&CORNER, &[]), 0);
}

#[test]
fn one_node_fixed_removes_the_translations_only() {
    assert_eq!(removed_modes(&CORNER, &all_of(&[0])), 3);
}

#[test]
fn two_nodes_fixed_leave_the_rotation_about_their_line() {
    assert_eq!(removed_modes(&CORNER, &all_of(&[0, 1])), 5);
}

#[test]
fn three_collinear_nodes_fixed_still_leave_the_rotation_about_their_line() {
    let line = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert_eq!(removed_modes(&line, &all_of(&[0, 1, 2])), 5);
}

#[test]
fn three_noncollinear_nodes_fixed_remove_every_mode() {
    assert_eq!(removed_modes(&CORNER, &all_of(&[0, 1, 2])), 6);
}

#[test]
fn a_face_fixed_removes_every_mode() {
    let cube: Vec<[f64; 3]> = (0..8)
        .map(|node| {
            [
                (node & 1) as f64,
                ((node >> 1) & 1) as f64,
                ((node >> 2) & 1) as f64,
            ]
        })
        .collect();
    let face: Vec<usize> = (0..8).filter(|node| node & 1 == 0).collect();
    assert_eq!(removed_modes(&cube, &all_of(&face)), 6);
}

#[test]
fn pinning_one_component_of_many_nodes_can_remove_every_mode() {
    let cube: Vec<[f64; 3]> = (0..8)
        .map(|node| {
            [
                (node & 1) as f64,
                ((node >> 1) & 1) as f64,
                ((node >> 2) & 1) as f64,
            ]
        })
        .collect();
    let dofs = [0, 6, 12, 18, 1, 2, 8];
    assert_eq!(removed_modes(&cube, &dofs), 6);
    assert_eq!(removed_modes(&cube, &dofs[..6]), 5);
}

#[test]
fn the_size_of_the_body_does_not_matter() {
    let scaled: Vec<[f64; 3]> = CORNER
        .iter()
        .map(|position| [1e6 * position[0], 1e6 * position[1], 1e6 * position[2]])
        .collect();
    assert_eq!(removed_modes(&scaled, &all_of(&[0, 1, 2])), 6);
    assert_eq!(removed_modes(&scaled, &all_of(&[0, 1])), 5);
}

#[test]
fn a_subdomain_with_nothing_constrained_has_every_rigid_mode_in_its_kernel() {
    assert_eq!(kernel(&CORNER, &[]).len(), 6);
}

#[test]
fn a_subdomain_pinned_at_one_node_keeps_its_rotations() {
    let pinned = all_of(&[0]);
    let modes = kernel(&CORNER, &pinned);
    assert_eq!(modes.len(), 3);
    modes.iter().for_each(|mode| {
        pinned
            .iter()
            .for_each(|&dof| assert!(mode[dof].abs() < 1e-12));
        assert!(mode.iter().any(|entry| entry.abs() > 1e-3));
    });
}

#[test]
fn a_fully_pinned_subdomain_has_no_kernel() {
    assert!(kernel(&CORNER, &all_of(&[0, 1, 2])).is_empty());
}

#[test]
fn kernel_pins_leave_the_kernel_nonsingular_on_them() {
    let modes = kernel(&CORNER, &[]);
    let free: Vec<usize> = (0..12).collect();
    let pins = kernel_pins(&modes, &free);
    assert_eq!(pins.len(), 6);
    let mut matrix: Vec<Vec<f64>> = pins
        .iter()
        .map(|&pin| modes.iter().map(|mode| mode[free[pin]]).collect())
        .collect();
    let mut determinant = 1.0;
    (0..6).for_each(|column| {
        let pivot = (column..6)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot);
        determinant *= matrix[column][column];
        (column + 1..6).for_each(|row| {
            let factor = matrix[row][column] / matrix[column][column];
            (column..6).for_each(|k| matrix[row][k] -= factor * matrix[column][k]);
        });
    });
    assert!(determinant.abs() > 1e-6);
}
