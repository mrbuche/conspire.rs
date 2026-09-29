use super::{Simplex, dot};
use crate::geometry::{
    Coordinate, Coordinates,
    mesh::{Connectivity, Mesh},
};

const TOLERANCE: f64 = 1e-12;

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < TOLERANCE, "{a} vs {b}");
}

#[test]
fn unit_triangle() {
    let s = Simplex::new([0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    assert_close(s.volume, 0.5);
    assert_eq!(s.gradients, [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]);
}

#[test]
fn unit_tetrahedron() {
    let s = Simplex::new(
        [0, 1, 2, 3],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    );
    assert_close(s.volume, 1.0 / 6.0);
    assert_eq!(
        s.gradients,
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    );
}

#[test]
fn triangle_embedded_in_space() {
    let (c, s) = (0.7f64.cos(), 0.7f64.sin());
    let place = |x: f64, y: f64| [x + 3.0, c * y - 1.0, s * y + 2.0];
    let simplex = Simplex::new(
        [0, 1, 2],
        [place(0.0, 0.0), place(2.0, 0.0), place(0.0, 3.0)],
    );
    assert_close(simplex.volume, 3.0);
    let field = |p: [f64; 3]| 2.0 * p[0] - p[1] + 0.5 * p[2];
    let values = [
        field(place(0.0, 0.0)),
        field(place(2.0, 0.0)),
        field(place(0.0, 3.0)),
    ];
    let gradient: [f64; 3] =
        std::array::from_fn(|k| (0..3).map(|a| values[a] * simplex.gradients[a][k]).sum());
    let normal = [0.0, -s, c];
    let full = [2.0, -1.0, 0.5];
    let along = dot(&full, &normal);
    (0..3).for_each(|k| assert_close(gradient[k], full[k] - along * normal[k]));
}

#[test]
fn gradients_sum_to_zero_and_reproduce_linear_fields() {
    let points = [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.4],
        [0.4, 1.5, 0.2],
        [0.3, 0.4, 1.7],
    ];
    let s = Simplex::new([0, 1, 2, 3], points);
    (0..3).for_each(|k| assert_close(s.gradients.iter().map(|g| g[k]).sum::<f64>(), 0.0));
    let slope = [1.5, -2.0, 0.25];
    let values: Vec<f64> = points.iter().map(|p| dot(&slope, p)).collect();
    (0..3).for_each(|k| {
        let g: f64 = (0..4).map(|a| values[a] * s.gradients[a][k]).sum();
        assert_close(g, slope[k]);
    });
}

fn mesh() -> Mesh<2> {
    Mesh::from((
        vec![
            Connectivity::Triangular(vec![[0usize, 1, 2]].into()),
            Connectivity::Triangular(vec![[1usize, 3, 2]].into()),
        ],
        Coordinates::from([
            Coordinate::from([0.0, 0.0]),
            Coordinate::from([1.0, 0.0]),
            Coordinate::from([0.0, 1.0]),
            Coordinate::from([1.0, 1.0]),
        ]),
    ))
}

#[test]
fn simplices_over_a_subset_of_elements() {
    let mesh = mesh();
    let all = mesh.simplices_over::<3>(&[0, 1]).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].nodes, [1, 3, 2]);
    let second = mesh.simplices_over::<3>(&[1]).unwrap();
    assert_eq!(second.len(), 1);
    assert_close(second[0].volume, 0.5);
    assert!(mesh.simplices_over::<3>(&[]).unwrap().is_empty());
}

#[test]
fn the_node_count_must_match_the_elements() {
    assert!(mesh().simplices_over::<4>(&[0]).is_none());
    let tetrahedron = Mesh::from((
        vec![Connectivity::Tetrahedral(vec![[0usize, 1, 2, 3]].into())],
        Coordinates::from([
            Coordinate::from([0.0, 0.0, 0.0]),
            Coordinate::from([1.0, 0.0, 0.0]),
            Coordinate::from([0.0, 1.0, 0.0]),
            Coordinate::from([0.0, 0.0, 1.0]),
        ]),
    ));
    assert!(tetrahedron.simplices_over::<3>(&[0]).is_none());
    assert_close(
        tetrahedron.simplices_over::<4>(&[0]).unwrap()[0].volume,
        1.0 / 6.0,
    );
}

#[test]
fn non_simplicial_meshes_have_no_simplices() {
    let quadrilateral = Mesh::from((
        vec![Connectivity::Quadrilateral(vec![[0usize, 1, 2, 3]].into())],
        Coordinates::from([
            Coordinate::from([0.0, 0.0]),
            Coordinate::from([1.0, 0.0]),
            Coordinate::from([1.0, 1.0]),
            Coordinate::from([0.0, 1.0]),
        ]),
    ));
    assert!(quadrilateral.simplices_over::<4>(&[0]).is_none());
}
