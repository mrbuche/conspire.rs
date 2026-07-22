use super::{Instance, conflicts};

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

#[test]
fn isolated_cell_costs_one() {
    let instance = Instance::new(vec![([0, 0], true)]);
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 1);
    assert!(instance.feasible(&assignment));
}

#[test]
fn two_adjacent_cells_cost_two() {
    let instance = Instance::new(vec![([0, 0], true), ([1, 0], true)]);
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 2);
    assert!(instance.feasible(&assignment));
}

#[test]
fn three_in_a_row_cannot_avoid_extra_cost() {
    let instance = Instance::new(vec![([0, 0], true), ([1, 0], true), ([2, 0], true)]);
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 3);
    assert!(instance.feasible(&assignment));
}

#[test]
fn unrequired_neighbor_is_never_assigned() {
    let instance = Instance::new(vec![([0, 0], true), ([1, 0], false)]);
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 1);
    assert!(instance.feasible(&assignment));
}

#[test]
fn solve_matches_bruteforce_on_examples() {
    for cells in [
        vec![([0, 0], true)],
        vec![([0, 0], true), ([1, 0], true)],
        vec![([0, 0], true), ([1, 0], true), ([2, 0], true)],
        vec![([0, 0], true), ([1, 0], false)],
    ] {
        let instance = Instance::new(cells);
        let (assignment, cost) = instance.solve();
        let (_, bruteforce_cost) = instance.solve_bruteforce();
        assert!(instance.feasible(&assignment));
        assert_eq!(cost, bruteforce_cost);
    }
}

fn fuzz_cells_2d(seed: u64) -> Vec<([i32; 2], bool)> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    (0..3)
        .flat_map(|x| (0..2).map(move |y| [x, y]))
        .map(|cell| (cell, rand() % 2 == 0))
        .collect()
}

#[test]
fn solve_matches_bruteforce_fuzz() {
    for seed in 0..200u64 {
        let cells = fuzz_cells_2d(seed);
        if !cells.iter().any(|(_, required)| *required) {
            continue;
        }
        let instance = Instance::new(cells);
        let (assignment, cost) = instance.solve();
        let (_, bruteforce_cost) = instance.solve_bruteforce();
        assert!(
            instance.feasible(&assignment),
            "seed {seed}: solve produced infeasible assignment"
        );
        assert_eq!(
            cost, bruteforce_cost,
            "seed {seed}: solve cost {cost} != bruteforce cost {bruteforce_cost}"
        );
    }
}
