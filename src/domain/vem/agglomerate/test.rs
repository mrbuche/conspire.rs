use super::Candidates;
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    vem::block::element::DEFAULT_STABILIZATION,
};

fn wedge(epsilon: f64) -> Mesh<3> {
    (
        vec![Connectivity::Tetrahedral(
            vec![[0, 1, 2, 3], [0, 2, 1, 4]].into(),
        )],
        Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, epsilon],
            [0.3, 0.3, -1.0],
        ]),
    )
        .into()
}

fn candidates(mesh: &Mesh<3>) -> Candidates {
    Candidates::new(mesh, 0.3, DEFAULT_STABILIZATION).unwrap()
}

#[test]
fn a_flat_tetrahedron_is_fast_and_its_agglomerate_is_not() {
    let (coarse, fine) = (candidates(&wedge(1.0e-1)), candidates(&wedge(1.0e-4)));
    let singles = |candidates: &Candidates| candidates.time_scale(&[0]).unwrap().value();
    let union = |candidates: &Candidates| candidates.time_scale(&[0, 1]).unwrap().value();
    assert!(singles(&fine) < 1.0e-2 * singles(&coarse));
    assert!(union(&fine) > 0.5 * union(&coarse));
    assert!(union(&fine) > 50.0 * singles(&fine));
}

#[test]
fn the_certificate_brackets_the_time_scale() {
    let candidates = candidates(&wedge(1.0e-1));
    let scale = candidates.time_scale(&[0, 1]).unwrap();
    assert!(
        candidates
            .time_scale_exceeds(&[0, 1], scale * 0.99)
            .unwrap()
    );
    assert!(
        !candidates
            .time_scale_exceeds(&[0, 1], scale * 1.01)
            .unwrap()
    );
}

#[test]
fn every_element_has_a_time_scale() {
    let scales = candidates(&wedge(1.0e-1)).time_scales().unwrap();
    assert_eq!(scales.len(), 2);
    assert!(scales.iter().all(|scale| scale.value() > 0.0));
}

#[test]
fn a_hexahedron_has_a_finite_time_scale() {
    let mesh = Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 3, 2, 4, 5, 7, 6]].into(),
        )],
        Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]),
    ));
    let scale = candidates(&mesh).time_scale(&[0]).unwrap().value();
    assert!(scale.is_finite() && scale > 0.0);
}

#[test]
fn elements_that_do_not_touch_cannot_be_joined() {
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(
            vec![[0, 1, 2, 3], [4, 5, 6, 7]].into(),
        )],
        Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
        ]),
    ));
    assert!(candidates(&mesh).time_scale(&[0, 1]).is_err());
}

fn tetrahedra(nodes: Vec<[f64; 3]>, blocks: Vec<Vec<[usize; 4]>>) -> Mesh<3> {
    (
        blocks
            .into_iter()
            .map(|block| Connectivity::Tetrahedral(block.into()))
            .collect::<Vec<_>>(),
        nodes.into(),
    )
        .into()
}

fn side_by_side(apex: [f64; 3], other: [f64; 3]) -> Mesh<3> {
    tetrahedra(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            apex,
            other,
        ],
        vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]],
    )
}

#[test]
fn a_wedge_is_a_valid_element() {
    assert_eq!(candidates(&wedge(1.0e-1)).check(&[0, 1], 0.01), Ok(()));
}

#[test]
fn a_single_element_is_valid() {
    assert_eq!(candidates(&wedge(1.0e-1)).check(&[0], 0.01), Ok(()));
}

#[test]
fn elements_in_different_blocks_are_not_joined() {
    let mesh = tetrahedra(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],
            [0.3, 0.3, -1.0],
        ],
        vec![vec![[0, 1, 2, 3]], vec![[0, 2, 1, 4]]],
    );
    assert_eq!(
        candidates(&mesh).check(&[0, 1], 0.01),
        Err("the elements are in different blocks".to_string())
    );
    assert_eq!(candidates(&mesh).check(&[1], 0.01), Ok(()));
}

#[test]
fn a_union_whose_mean_is_outside_is_not_star_shaped() {
    let mesh = side_by_side([2.0, 2.0, 1.0], [2.0, 2.0, -1.0]);
    assert_eq!(
        candidates(&mesh).check(&[0, 1], 0.01),
        Err("the element is not star-shaped about the mean of its nodes".to_string())
    );
}

#[test]
fn the_volume_floor_rejects_a_small_sub_tetrahedron() {
    let mesh = side_by_side([0.3, 0.3, 1.0], [0.3, 0.3, -1.0]);
    assert!(candidates(&mesh).check(&[0, 1], 0.01).is_ok());
    assert!(candidates(&mesh).check(&[0, 1], 100.0).is_err());
}
