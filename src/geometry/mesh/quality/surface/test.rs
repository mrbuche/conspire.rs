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

#[test]
fn open_quad_has_one_boundary_loop() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3]], 4);
    let loops = mesh.boundary_loops();
    assert_eq!(loops.len(), 1);
    let mut nodes = loops[0].clone();
    assert_eq!(nodes.len(), 4, "four unique nodes around the loop");
    nodes.sort_unstable();
    assert_eq!(nodes, vec![0, 1, 2, 3]);
}

#[test]
fn closed_tetrahedron_has_no_boundary_loops() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], 4);
    assert!(mesh.boundary_loops().is_empty());
}
