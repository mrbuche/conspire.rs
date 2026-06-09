use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn quad<const D: usize>(coordinates: Vec<[f64; D]>) -> Mesh<D> {
    let connectivities = vec![Connectivity::Quadrilateral(vec![[0, 1, 2, 3]].into())];
    Mesh::from((connectivities, coordinates.into()))
}

#[test]
fn unit_square_is_perfect() {
    let mesh = quad(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    assert_eq!(mesh.minimum_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![1.0]]);
}

#[test]
fn stretched_rectangle_edge_ratio_is_longest_over_shortest() {
    let mesh = quad(vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]]);
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![3.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
}

#[test]
fn inverted_quad_is_negative() {
    let mesh = quad(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
}

#[test]
fn embedded_square_in_three_dimensions_is_perfect() {
    let mesh = quad(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.maximum_edge_ratios(), vec![vec![1.0]]);
}
