use crate::geometry::mesh::{Boundary, Connectivity, ElementsFaces, Mesh, PrimitiveConnectivity};

fn collected(faces: &impl ElementsFaces, element: usize) -> Vec<Vec<usize>> {
    faces
        .element_faces(element)
        .map(|face| face.as_ref().to_vec())
        .collect()
}

fn tetrahedra() -> Vec<[usize; 4]> {
    vec![[0, 1, 2, 3], [0, 2, 1, 4]]
}

fn sorted(mut faces: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    faces.iter_mut().for_each(|face| face.sort_unstable());
    faces.sort();
    faces
}

#[test]
fn fixed_tetrahedra_give_the_faces_of_the_connectivity() {
    let fixed = PrimitiveConnectivity::<3, 4>::from(tetrahedra());
    let dynamic = Connectivity::Tetrahedral(tetrahedra().into());
    assert_eq!(fixed.number_of_elements(), 2);
    for (element, nodes) in tetrahedra().iter().enumerate() {
        assert_eq!(collected(&fixed, element), dynamic.element_faces(nodes));
    }
}

#[test]
fn fixed_elements_keep_the_sizes_of_their_faces() {
    let sizes = |faces: Vec<Vec<usize>>| faces.iter().map(|face| face.len()).collect::<Vec<_>>();
    assert_eq!(
        sizes(collected(
            &PrimitiveConnectivity::<3, 8>::from(vec![[0, 1, 2, 3, 4, 5, 6, 7]]),
            0
        )),
        [4; 6]
    );
    assert_eq!(
        sizes(collected(
            &PrimitiveConnectivity::<3, 5>::from(vec![[0, 1, 2, 3, 4]]),
            0
        )),
        [3, 3, 3, 3, 4]
    );
    assert_eq!(
        sizes(collected(
            &PrimitiveConnectivity::<3, 6>::from(vec![[0, 1, 2, 3, 4, 5]]),
            0
        )),
        [4, 4, 4, 3, 3]
    );
}

#[test]
fn a_boundary_of_a_connectivity_matches_a_boundary_of_a_mesh() {
    let connectivity = PrimitiveConnectivity::<3, 4>::from(tetrahedra());
    let fixed = Boundary::new(&connectivity);
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(tetrahedra().into())],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],
            [0.3, 0.3, -1.0],
        ]
        .into(),
    ));
    let dynamic = Boundary::try_from(&mesh).unwrap();
    assert_eq!(fixed.number_of_elements(), dynamic.number_of_elements());
    assert_eq!(fixed.adjacent(), dynamic.adjacent());
    for elements in [vec![0], vec![1], vec![0, 1]] {
        assert_eq!(
            sorted(fixed.faces(&elements).unwrap()),
            sorted(dynamic.faces(&elements).unwrap())
        );
    }
    assert!(fixed.surface(&[0, 1]).unwrap().is_sphere());
}

#[test]
fn fixed_elements_that_share_a_face_are_adjacent() {
    let connectivity = PrimitiveConnectivity::<3, 4>::from(tetrahedra());
    assert_eq!(
        Boundary::new(&connectivity).adjacent(),
        vec![vec![1], vec![0]]
    );
}
