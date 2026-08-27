use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
        solid::elastic::ElasticElements,
    },
    units::Stress,
};

#[test]
fn block_error_carries_no_escape_codes() {
    let reference = NodalReferenceCoordinates::<3>::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let block = Block::<_, Tetrahedron, 1, 3, 4, 4>::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        vec![[0, 1, 2, 3]],
        &reference,
    ));
    let coordinates = NodalCoordinates::<3>::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]);
    let error = block.nodal_forces(&coordinates).unwrap_err();
    let rendered = format!("{error}");
    assert!(!rendered.contains('\u{1b}'))
}
