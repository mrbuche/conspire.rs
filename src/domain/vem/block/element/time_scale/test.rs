use super::{fastest_time_scale, time_scale_exceeds};
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::block::element::solid::elastic::ElasticElement,
    math::Quantity,
    mechanics::ReferenceCoordinate,
    units::{Density, Stress},
    vem::{
        NodalCoordinates,
        block::element::{
            Element, ElementNodalReferenceCoordinates, mass::LumpedMassVirtualElement,
        },
    },
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn material() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(13.0e9),
        shear_modulus: Stress::pascals(3.0e9),
    }
}

fn certified_around_the_time_scale(nodes: &[[f64; 3]], faces: &[Vec<usize>]) {
    let coordinates: ElementNodalReferenceCoordinates = faces
        .iter()
        .map(|face| {
            face.iter()
                .map(|&node| ReferenceCoordinate::from(nodes[node]))
                .collect()
        })
        .collect();
    let faces_indices = (0..faces.len()).collect::<Vec<_>>();
    let nodes_indices = (0..nodes.len()).collect::<Vec<_>>();
    let element = Element::from((coordinates, &faces_indices[..], &nodes_indices[..], faces));
    let reference = crate::vem::NodalReferenceCoordinates::from(nodes.to_vec());
    let stiffnesses = element
        .nodal_stiffnesses(&material(), &NodalCoordinates::from(nodes.to_vec()))
        .unwrap();
    let masses = element.nodal_lumped_masses(DENSITY, &reference);
    let scale = fastest_time_scale(&stiffnesses, &masses);
    assert!(
        time_scale_exceeds(&stiffnesses, &masses, scale * 0.99),
        "{scale:?}"
    );
    assert!(
        !time_scale_exceeds(&stiffnesses, &masses, scale * 1.01),
        "{scale:?}"
    );
}

fn tetrahedron_faces() -> Vec<Vec<usize>> {
    vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]]
}

fn bipyramid_faces() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
        vec![1, 0, 4],
        vec![2, 1, 4],
        vec![0, 2, 4],
    ]
}

#[test]
fn brackets_a_tetrahedron() {
    certified_around_the_time_scale(
        &[
            [0.1, 0.2, 0.0],
            [1.3, 0.1, 0.2],
            [0.2, 0.9, 0.1],
            [0.3, 0.4, 1.2],
        ],
        &tetrahedron_faces(),
    );
}

#[test]
fn brackets_a_box() {
    certified_around_the_time_scale(
        &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ],
        &[
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![3, 7, 6, 2],
            vec![0, 4, 7, 3],
            vec![1, 2, 6, 5],
        ],
    );
}

#[test]
fn brackets_a_lopsided_bipyramid() {
    let s = 3.0_f64.sqrt() / 2.0;
    certified_around_the_time_scale(
        &[
            [1.0, 0.0, 0.0],
            [-0.5, s, 0.0],
            [-0.5, -s, 0.0],
            [0.1, 0.2, 0.9],
            [-0.2, 0.1, -0.4],
        ],
        &bipyramid_faces(),
    );
}

#[test]
fn brackets_an_agglomerated_flat_wedge() {
    certified_around_the_time_scale(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1e-5],
            [0.3, 0.3, -1.0],
        ],
        &bipyramid_faces(),
    );
}

#[test]
fn brackets_a_sliver_tetrahedron() {
    certified_around_the_time_scale(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1e-5],
        ],
        &tetrahedron_faces(),
    );
}
