use crate::geometry::mesh::{Connectivities, Connectivity, SideSet};

fn hexahedra() -> Connectivities {
    vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]].into(),
    )]
    .into()
}

#[test]
fn the_faces_of_a_side_set_are_those_of_the_elements_it_names() {
    let faces = hexahedra().faces_from(&SideSet::new(vec![(0, 5), (1, 5)]));
    assert_eq!(faces, Some(vec![vec![4, 5, 6, 7], vec![8, 9, 10, 11]]));
}

#[test]
fn the_faces_come_in_the_order_of_the_sides() {
    let faces = hexahedra().faces_from(&SideSet::new(vec![(1, 4), (0, 5)]));
    assert_eq!(faces, Some(vec![vec![4, 7, 6, 5], vec![4, 5, 6, 7]]));
}

#[test]
fn an_element_may_have_several_faces_in_the_set() {
    let faces = hexahedra().faces_from(&SideSet::new(vec![(0, 5), (0, 4)]));
    assert_eq!(faces, Some(vec![vec![4, 5, 6, 7], vec![0, 3, 2, 1]]));
}

#[test]
fn the_faces_of_tetrahedra_have_three_nodes() {
    let connectivities = Connectivities::from(vec![Connectivity::Tetrahedral(
        vec![[0, 1, 2, 3], [1, 2, 3, 4]].into(),
    )]);
    let faces = connectivities.faces_from(&SideSet::new(vec![(1, 3)]));
    assert_eq!(faces, Some(vec![vec![1, 3, 2]]));
}

#[test]
fn the_elements_of_several_blocks_are_counted_together() {
    let connectivities = Connectivities::from(vec![
        Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        Connectivity::Tetrahedral(vec![[4, 5, 6, 8]].into()),
    ]);
    let faces = connectivities.faces_from(&SideSet::new(vec![(1, 0)]));
    assert_eq!(faces, Some(vec![vec![4, 5, 8]]));
}

#[test]
fn an_element_the_blocks_do_not_have_gives_no_faces() {
    assert_eq!(hexahedra().faces_from(&SideSet::new(vec![(2, 0)])), None);
}

#[test]
fn a_face_the_element_does_not_have_gives_no_faces() {
    assert_eq!(hexahedra().faces_from(&SideSet::new(vec![(0, 6)])), None);
}

#[test]
fn an_empty_set_has_no_faces_to_give() {
    assert_eq!(
        hexahedra().faces_from(&SideSet::new(Vec::new())),
        Some(Vec::new())
    );
}
