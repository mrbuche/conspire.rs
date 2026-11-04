use super::Surface;

fn surface(faces: &[Vec<usize>]) -> Result<Surface, String> {
    Surface::try_from(faces)
}

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

fn torus() -> Vec<Vec<usize>> {
    let node = |i: usize, j: usize| 3 * (i % 3) + j % 3;
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            vec![
                node(i, j),
                node(i + 1, j),
                node(i + 1, j + 1),
                node(i, j + 1),
            ]
        })
        .collect()
}

#[test]
fn a_tetrahedron_and_a_cube_are_spheres() {
    for faces in [tetrahedron([0, 1, 2, 3]), cube()] {
        let surface = surface(&faces).unwrap();
        assert!(surface.is_sphere());
        assert_eq!(surface.number_of_components(), 1);
        assert_eq!(surface.euler_characteristic(), 2);
        assert_eq!(surface.genera(), vec![0]);
    }
}

#[test]
fn a_torus_has_one_handle() {
    let surface = surface(&torus()).unwrap();
    assert!(!surface.is_sphere());
    assert_eq!(surface.number_of_components(), 1);
    assert_eq!(surface.euler_characteristic(), 0);
    assert_eq!(surface.genera(), vec![1]);
}

#[test]
fn disjoint_surfaces_are_counted_separately() {
    let mut faces = tetrahedron([0, 1, 2, 3]);
    faces.extend(tetrahedron([4, 5, 6, 7]));
    let surface = surface(&faces).unwrap();
    assert!(!surface.is_sphere());
    assert_eq!(surface.number_of_components(), 2);
    assert_eq!(surface.euler_characteristics(), [2, 2]);
    assert_eq!(surface.euler_characteristic(), 4);
    assert_eq!(surface.genera(), vec![0, 0]);
}

#[test]
fn components_have_their_own_genus() {
    let mut faces = tetrahedron([100, 101, 102, 103]);
    faces.extend(torus());
    let surface = surface(&faces).unwrap();
    assert_eq!(surface.euler_characteristics(), [2, 0]);
    assert_eq!(surface.genera(), vec![0, 1]);
}

#[test]
fn a_sphere_is_ensured() {
    assert_eq!(surface(&cube()).unwrap().ensure_sphere(), Ok(()));
}

#[test]
fn a_torus_is_not_ensured_and_has_genus_one() {
    assert_eq!(
        surface(&torus()).unwrap().ensure_sphere(),
        Err("the surface is not a sphere, it has genus 1".to_string())
    );
}

#[test]
fn several_components_are_not_ensured() {
    let mut faces = tetrahedron([0, 1, 2, 3]);
    faces.extend(tetrahedron([4, 5, 6, 7]));
    assert_eq!(
        surface(&faces).unwrap().ensure_sphere(),
        Err("the surface has 2 components, not one".to_string())
    );
}

#[test]
fn a_missing_face_is_not_closed() {
    let mut faces = cube();
    faces.pop();
    assert_eq!(
        surface(&faces).err(),
        Some("the surface is not closed".to_string())
    );
}

#[test]
fn a_reversed_face_shares_an_edge_direction() {
    let mut faces = cube();
    faces[0].reverse();
    assert_eq!(
        surface(&faces).err(),
        Some("an edge is used twice in the same direction".to_string())
    );
}

#[test]
fn two_tetrahedra_meeting_at_a_node_are_pinched() {
    let mut faces = tetrahedron([0, 1, 2, 3]);
    faces.extend(tetrahedron([0, 4, 5, 6]));
    assert_eq!(surface(&faces).err(), Some("node 0 is pinched".to_string()));
}

#[test]
fn no_faces_are_an_error() {
    assert_eq!(surface(&[]).err(), Some("there are no faces".to_string()));
}

#[test]
fn a_face_needs_three_nodes() {
    let mut faces = cube();
    faces[0] = vec![0, 2];
    assert_eq!(
        surface(&faces).err(),
        Some("a face has fewer than three nodes".to_string())
    );
}
