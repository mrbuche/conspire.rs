use super::Boundary;
use crate::geometry::mesh::{Connectivity, Mesh};

fn tetrahedra(elements: Vec<[usize; 4]>) -> Mesh<3> {
    (
        vec![Connectivity::Tetrahedral(elements.into())],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],
            [0.3, 0.3, -1.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
        ]
        .into(),
    )
        .into()
}

fn boundary() -> Boundary {
    Boundary::try_from(&tetrahedra(vec![[0, 1, 2, 3], [0, 2, 1, 4], [5, 6, 7, 8]])).unwrap()
}

#[test]
fn one_element_has_all_of_its_faces() {
    let boundary = boundary();
    assert_eq!(boundary.number_of_elements(), 3);
    assert_eq!(boundary.faces(&[0]).unwrap().len(), 4);
}

#[test]
fn elements_sharing_a_face_lose_it() {
    let faces = boundary().faces(&[0, 1]).unwrap();
    assert_eq!(faces.len(), 6);
    assert!(faces.iter().all(|face| face.iter().any(|&node| node > 2)));
}

#[test]
fn the_union_of_two_tetrahedra_is_a_sphere() {
    let surface = boundary().surface(&[0, 1]).unwrap();
    assert!(surface.is_sphere());
}

#[test]
fn the_order_of_the_elements_does_not_matter() {
    let boundary = boundary();
    let mut forward = boundary.faces(&[0, 1]).unwrap();
    let mut backward = boundary.faces(&[1, 0]).unwrap();
    forward.iter_mut().for_each(|face| face.sort_unstable());
    backward.iter_mut().for_each(|face| face.sort_unstable());
    forward.sort();
    backward.sort();
    assert_eq!(forward, backward);
}

#[test]
fn elements_that_do_not_share_a_face_are_not_connected() {
    assert_eq!(
        boundary().faces(&[0, 2]),
        Err("the elements are not connected through shared faces")
    );
}

#[test]
fn no_elements_are_an_error() {
    assert_eq!(boundary().faces(&[]), Err("there are no elements"));
}

#[test]
fn surface_elements_have_no_boundary() {
    let mesh = Mesh::from((
        vec![Connectivity::Triangular(vec![[0, 1, 2]].into())],
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into(),
    ));
    assert_eq!(
        Boundary::try_from(&mesh).err(),
        Some("a boundary requires three-dimensional elements")
    );
}
