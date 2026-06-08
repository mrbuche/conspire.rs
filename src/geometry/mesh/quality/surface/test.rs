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

#[test]
fn fin_has_one_non_manifold_edge() {
    let mesh = triangles(vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]], 5);
    let edges = mesh.non_manifold_edges();
    assert_eq!(edges.len(), 1);
    let mut edge = edges[0];
    edge.sort_unstable();
    assert_eq!(edge, [0, 1]);
}

#[test]
fn manifold_mesh_has_no_non_manifold_edges() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], 4);
    assert!(mesh.non_manifold_edges().is_empty());
}

#[test]
fn adjacent_fins_form_one_seam() {
    let mesh = triangles(
        vec![
            [0, 1, 3],
            [0, 1, 4],
            [0, 1, 5],
            [1, 2, 6],
            [1, 2, 7],
            [1, 2, 8],
        ],
        9,
    );
    let seams = mesh.non_manifold_seams();
    assert_eq!(seams.len(), 1);
    assert_eq!(seams[0].len(), 2);
}

#[test]
fn disjoint_fins_form_two_seams() {
    let mesh = triangles(
        vec![
            [0, 1, 4],
            [0, 1, 5],
            [0, 1, 6],
            [2, 3, 7],
            [2, 3, 8],
            [2, 3, 9],
        ],
        10,
    );
    assert_eq!(mesh.non_manifold_seams().len(), 2);
}

#[test]
fn bowtie_vertex_is_non_manifold() {
    let mesh = triangles(vec![[0, 1, 2], [0, 3, 4]], 5);
    assert!(mesh.non_manifold_edges().is_empty());
    assert_eq!(mesh.non_manifold_vertices(), vec![0]);
}

#[test]
fn tetrahedron_vertices_are_manifold() {
    let mesh = triangles(vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], 4);
    assert!(mesh.non_manifold_vertices().is_empty());
}
