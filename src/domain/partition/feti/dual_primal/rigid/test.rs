use super::removed_modes;

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
