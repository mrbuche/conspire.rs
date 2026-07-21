use crate::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh},
};

fn mesh(connectivity: Vec<[usize; 3]>, points: Vec<[f64; 3]>) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Triangular(connectivity.into())],
        Coordinates::from(points),
    ))
}

#[test]
fn piercing_triangles_intersect() {
    let mesh = mesh(
        vec![[0, 1, 2], [3, 4, 5]],
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.5, -1.0],
            [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.0],
        ],
    );
    assert_eq!(mesh.self_intersections(), vec![[0, 1]]);
}

#[test]
fn separated_triangles_do_not_intersect() {
    let mesh = mesh(
        vec![[0, 1, 2], [3, 4, 5]],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
    );
    assert!(mesh.self_intersections().is_empty());
}

#[test]
fn adjacent_triangles_are_not_self_intersections() {
    let mesh = mesh(
        vec![[0, 1, 2], [0, 1, 3]],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.5],
        ],
    );
    assert!(mesh.self_intersections().is_empty());
}
