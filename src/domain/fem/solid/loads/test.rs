use super::traction_forces;
use crate::{
    fem::NodalReferenceCoordinates,
    geometry::mesh::SideSet,
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::Force,
};

fn coordinates() -> NodalReferenceCoordinates<3> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    .into()
}

fn side_set(faces: Vec<Vec<usize>>) -> SideSet {
    SideSet::new((0..faces.len()).map(|element| (element, 0)).collect()).with_faces(faces)
}

#[test]
fn a_quadrilateral_gives_a_quarter_of_its_load_to_each_node() -> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3]]),
        &[0.0, 0.0, 8.0].into(),
    )
    .unwrap();
    Assert::default().eq_within_tols(
        &forces,
        &[
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0; 3],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0; 3],
        ]
        .into(),
    )
}

#[test]
fn a_node_shared_by_faces_gets_the_load_of_each() -> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4]]),
        &[0.0, 0.0, 8.0].into(),
    )
    .unwrap();
    Assert::default().eq_within_tols(
        &forces,
        &[
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
        ]
        .into(),
    )
}

#[test]
fn triangles_and_quadrilaterals_together_carry_the_traction_on_the_area()
-> Result<(), AssertionError> {
    let forces = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 4, 3], vec![1, 2, 5], vec![1, 5, 4]]),
        &[1.0, 2.0, 3.0].into(),
    )
    .unwrap();
    let total = forces.iter().cloned().sum::<Force>();
    Assert::default().eq_within_tols(&total, &[2.0, 4.0, 6.0].into())
}

#[test]
fn a_side_set_without_faces_is_an_error() {
    let error = traction_forces(
        &coordinates(),
        &SideSet::new(vec![(0, 0)]),
        &[0.0, 0.0, 1.0].into(),
    )
    .unwrap_err();
    assert!(error.contains("no faces"));
}

#[test]
fn a_face_that_is_not_a_triangle_or_a_quadrilateral_is_an_error() {
    let error = traction_forces(
        &coordinates(),
        &side_set(vec![vec![0, 1, 2, 5, 4]]),
        &[0.0, 0.0, 1.0].into(),
    )
    .unwrap_err();
    assert!(error.contains("5 nodes"));
}
