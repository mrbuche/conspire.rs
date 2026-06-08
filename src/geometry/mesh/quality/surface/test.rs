use crate::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh},
};

fn triangles(connectivity: Vec<[usize; 3]>, points: usize) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Triangular(connectivity.into())],
        Coordinates::from(
            (0..points)
                .map(|i| [i as f64, 0.0, 0.0])
                .collect::<Vec<_>>(),
        ),
    ))
}

#[test]
fn open_quad_has_four_boundary_edges() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3]], 4);
    assert_eq!(mesh.boundary_edges().len(), 4);
}

#[test]
fn closed_tetrahedron_is_watertight() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], 4);
    assert!(mesh.boundary_edges().is_empty());
}
