use super::{BoundaryConditions, select_corners};
use crate::{domain::feti::dual_primal::rigid::removed_modes, geometry::mesh::Partition};

const LATTICE: [usize; 3] = [5, 4, 3];

fn node(i: usize, j: usize, k: usize) -> usize {
    i + LATTICE[0] * (j + LATTICE[1] * k)
}

fn positions() -> Vec<[f64; 3]> {
    (0..LATTICE[2])
        .flat_map(|k| {
            (0..LATTICE[1])
                .flat_map(move |j| (0..LATTICE[0]).map(move |i| [i as f64, j as f64, k as f64]))
        })
        .collect()
}

fn one_part() -> Partition {
    Partition::from_parts_nodes(vec![(0..LATTICE.iter().product()).collect()])
}

#[test]
fn a_clamped_face_makes_only_enough_corners_to_remove_the_rigid_modes() {
    let face: Vec<usize> = (0..LATTICE[2])
        .flat_map(|k| (0..LATTICE[1]).map(move |j| node(0, j, k)))
        .collect();
    let conditions = face
        .iter()
        .flat_map(|&node| (0..3).map(move |component| (node, component)))
        .fold(
            BoundaryConditions::none(),
            |conditions, (node, component)| conditions.prescribed(node, component, 0.0),
        );
    let positions = positions();
    let corners = select_corners(&one_part(), &positions, &conditions, &conditions.rows());
    let chosen = corners.nodes();
    assert!(chosen.len() < face.len(), "{} corners", chosen.len());
    assert!(chosen.iter().all(|node| face.contains(node)));
    let held: Vec<usize> = chosen
        .iter()
        .flat_map(|&node| (0..3).map(move |component| 3 * node + component))
        .collect();
    assert_eq!(removed_modes(&positions, &held), 6);
}

#[test]
fn a_constraint_of_several_nodes_makes_all_of_them_corners() {
    let conditions = BoundaryConditions::none().linear(vec![(3, 0, 1.0), (7, 0, -1.0)], 0.0);
    let corners = select_corners(&one_part(), &positions(), &conditions, &conditions.rows());
    assert_eq!(corners.nodes(), [3, 7]);
}
