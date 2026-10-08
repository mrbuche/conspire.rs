use super::sphere;

fn tetrahedron(nodes: [usize; 4]) -> Vec<Vec<usize>> {
    let [a, b, c, d] = nodes;
    vec![vec![a, c, b], vec![a, b, d], vec![a, d, c], vec![b, c, d]]
}

fn cube() -> Vec<Vec<usize>> {
    vec![
        vec![0, 2, 3, 1],
        vec![4, 5, 7, 6],
        vec![0, 1, 5, 4],
        vec![2, 6, 7, 3],
        vec![0, 4, 6, 2],
        vec![1, 3, 7, 5],
    ]
}

#[test]
fn a_tetrahedron_and_a_cube_are_spheres() {
    assert_eq!(sphere(&tetrahedron([0, 1, 2, 3])), Ok(()));
    assert_eq!(sphere(&cube()), Ok(()));
}

#[test]
fn a_missing_face_is_not_closed() {
    let mut faces = cube();
    faces.pop();
    assert_eq!(sphere(&faces), Err("the surface is not closed".to_string()));
}

#[test]
fn a_reversed_face_shares_an_edge_direction() {
    let mut faces = cube();
    faces[0].reverse();
    assert_eq!(
        sphere(&faces),
        Err("an edge is used twice in the same direction".to_string())
    );
}

#[test]
fn two_tetrahedra_meeting_at_a_node_are_pinched() {
    let mut faces = tetrahedron([0, 1, 2, 3]);
    faces.extend(tetrahedron([0, 4, 5, 6]));
    assert_eq!(sphere(&faces), Err("node 0 is pinched".to_string()));
}

#[test]
fn disjoint_surfaces_are_several_components() {
    let mut faces = tetrahedron([0, 1, 2, 3]);
    faces.extend(tetrahedron([4, 5, 6, 7]));
    assert_eq!(
        sphere(&faces),
        Err("the surface has several components".to_string())
    );
}

#[test]
fn a_torus_is_not_a_sphere() {
    let node = |i: usize, j: usize| 3 * (i % 3) + j % 3;
    let faces = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            vec![
                node(i, j),
                node(i + 1, j),
                node(i + 1, j + 1),
                node(i, j + 1),
            ]
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sphere(&faces),
        Err("the surface is not a sphere".to_string())
    );
}
