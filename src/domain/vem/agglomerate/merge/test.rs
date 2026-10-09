use super::{Agglomeration, Reference};
use crate::{
    geometry::mesh::{Connectivity, Merging, Mesh, PrimitiveConnectivity},
    units::Time,
    vem::{agglomerate::Candidates, block::element::DEFAULT_STABILIZATION},
};

fn wedge(epsilon: f64, blocks: Vec<Vec<[usize; 4]>>) -> Mesh<3> {
    (
        blocks
            .into_iter()
            .map(|block| Connectivity::Tetrahedral(block.into()))
            .collect::<Vec<_>>(),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, epsilon],
            [0.3, 0.3, -1.0],
        ]
        .into(),
    )
        .into()
}

fn candidates(mesh: &Mesh<3>) -> Candidates {
    Candidates::from_mesh(mesh, 0.3, DEFAULT_STABILIZATION).unwrap()
}

fn agglomeration(reference: f64) -> Agglomeration {
    Agglomeration {
        reference: Reference::Value(Time::seconds(reference)),
        step_reduction: 2.0,
        minimum_volume: 0.01,
        merging: Merging {
            minimum_improvement: 1.2,
            passes: 5,
        },
    }
}

fn union_scale(mesh: &Mesh<3>) -> f64 {
    candidates(mesh).time_scale(&[0, 1]).unwrap().value()
}

#[test]
fn elements_of_one_topology_are_joined_like_the_elements_of_a_mesh() {
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let connectivity = PrimitiveConnectivity::<3, 4>::from(vec![[0, 1, 2, 3], [0, 2, 1, 4]]);
    let typed = Candidates::new(
        &connectivity,
        mesh.coordinates().clone(),
        0.3,
        DEFAULT_STABILIZATION,
    );
    let agglomeration = agglomeration(union_scale(&mesh));
    let from_mesh = candidates(&mesh).agglomerate(&agglomeration).unwrap();
    let from_connectivity = typed.agglomerate(&agglomeration).unwrap();
    assert_eq!(from_connectivity.elements_parts, [0, 0]);
    assert_eq!(from_connectivity.elements_parts, from_mesh.elements_parts);
    assert_eq!(from_connectivity.unresolved, from_mesh.unresolved);
    assert_eq!(
        from_connectivity.time_scales[0].value(),
        from_mesh.time_scales[0].value()
    );
}

#[test]
fn a_fixed_topology_has_its_elements_in_one_block() {
    let connectivity = PrimitiveConnectivity::<3, 4>::from(vec![[0, 1, 2, 3], [0, 2, 1, 4]]);
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let typed = Candidates::new(
        &connectivity,
        mesh.coordinates().clone(),
        0.3,
        DEFAULT_STABILIZATION,
    );
    assert_eq!(typed.check(&[0, 1], 0.01), Ok(()));
}

#[test]
fn a_flat_wedge_is_joined_into_one_element() {
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let result = candidates(&mesh)
        .agglomerate(&agglomeration(union_scale(&mesh)))
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 0]);
    assert!(result.unresolved.is_empty());
    assert_eq!(result.time_scales.len(), 1);
    assert_eq!(result.mesh(&mesh).unwrap().number_of_elements(), 1);
}

#[test]
fn elements_that_are_fast_enough_are_left_alone() {
    let mesh = wedge(1.0e-1, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let reference = union_scale(&mesh);
    let result = candidates(&mesh)
        .agglomerate(&Agglomeration {
            step_reduction: 1.0e6,
            ..agglomeration(reference)
        })
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 1]);
    assert!(result.unresolved.is_empty());
}

#[test]
fn elements_in_different_blocks_stay_unresolved() {
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3]], vec![[0, 2, 1, 4]]]);
    let result = candidates(&mesh)
        .agglomerate(&agglomeration(union_scale(&mesh)))
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 1]);
    assert_eq!(result.unresolved, vec![0]);
}

#[test]
fn no_passes_change_nothing() {
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let result = candidates(&mesh)
        .agglomerate(&Agglomeration {
            merging: Merging {
                minimum_improvement: 1.2,
                passes: 0,
            },
            ..agglomeration(union_scale(&mesh))
        })
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 1]);
    assert_eq!(result.unresolved, vec![0]);
}

#[test]
fn an_unreachable_improvement_prevents_the_join() {
    let mesh = wedge(1.0e-4, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let result = candidates(&mesh)
        .agglomerate(&Agglomeration {
            merging: Merging {
                minimum_improvement: 1.0e12,
                passes: 5,
            },
            ..agglomeration(union_scale(&mesh))
        })
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 1]);
}

#[test]
fn the_median_reference_is_one_of_the_element_time_scales() {
    let mesh = wedge(1.0e-1, vec![vec![[0, 1, 2, 3], [0, 2, 1, 4]]]);
    let candidates = candidates(&mesh);
    let result = candidates
        .agglomerate(&Agglomeration {
            reference: Reference::Median,
            ..agglomeration(1.0)
        })
        .unwrap();
    let scales = candidates.time_scales().unwrap();
    assert!(
        scales
            .iter()
            .any(|scale| scale.value() == result.reference.value())
    );
}

#[test]
fn good_hexahedra_are_left_alone() {
    let mesh = (
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]].into(),
        )],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
        .into(),
    )
        .into();
    let result = candidates(&mesh)
        .agglomerate(&Agglomeration {
            reference: Reference::Median,
            ..agglomeration(1.0)
        })
        .unwrap();
    assert_eq!(result.elements_parts, vec![0, 1]);
    assert!(result.unresolved.is_empty());
}
