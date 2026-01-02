use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block,
            element::{
                quadratic::{Pyramid, Tetrahedron, Wedge},
                serendipity::Hexahedron,
            },
        },
        solid::elastic::ElasticFiniteElementModel,
    },
    math::TensorVec,
};

const M: usize = 3;

#[test]
fn f3d() {
    let constitutive = NeoHookean {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    };
    let hexahedra = Block::<_, Hexahedron, 27, M, 20>::from((
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    ));
    let pyramids = Block::<_, Pyramid, 27, M, 13>::from((
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    ));
    let tetrahedra = Block::<_, Tetrahedron, 4, M, 10>::from((
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    ));
    let wedges = Block::<_, Wedge, 18, M, 15>::from((
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    ));
    let blocks = Blocks(hexahedra, Blocks(pyramids, Blocks(tetrahedra, wedges)));
    let model = Model {
        blocks,
        coordinates: NodalReferenceCoordinates::new(),
    };
    model.nodal_forces(&NodalCoordinates::new()).unwrap();
    todo!()
}
