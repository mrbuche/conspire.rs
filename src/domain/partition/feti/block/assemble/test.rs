use super::local_stiffness_and_force;
use crate::{
    constitutive::solid::{
        elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        hyperelastic::NeoHookean,
    },
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
    },
    math::Tensor,
};

fn block() -> Block<NeoHookean, Tetrahedron, 1, 3, 4, 4> {
    let reference_coordinates = NodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    Block::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        vec![[0, 1, 2, 3]],
        &reference_coordinates,
    ))
}

fn undeformed_coordinates() -> NodalCoordinates<3> {
    NodalCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

#[test]
fn zero_force_and_symmetric_stiffness_at_zero_deformation() {
    let block = block();
    let coordinates = undeformed_coordinates();
    let (stiffness, force) =
        local_stiffness_and_force(&block, &coordinates, &[0, 1, 2, 3]).unwrap();
    force.iter().for_each(|&entry| assert!(entry.abs() < 1e-10));
    (0..stiffness.len()).for_each(|row| {
        (0..stiffness.len()).for_each(|column| {
            assert!((stiffness[row][column] - stiffness[column][row]).abs() < 1e-8);
        })
    });
}

#[test]
fn an_element_missing_from_the_subdomain_contributes_nothing() {
    let block = block();
    let coordinates = undeformed_coordinates();
    let (stiffness, force) = local_stiffness_and_force(&block, &coordinates, &[0, 1, 2]).unwrap();
    assert_eq!(force.len(), 9);
    force.iter().for_each(|&entry| assert_eq!(entry, 0.0));
    (0..9).for_each(|row| {
        (0..9).for_each(|column| assert_eq!(stiffness[row][column], 0.0));
    });
}
