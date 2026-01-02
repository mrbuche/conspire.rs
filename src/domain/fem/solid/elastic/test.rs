use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block, FiniteElementBlock,
            element::{
                quadratic::{Pyramid, Tetrahedron, Wedge},
                serendipity::Hexahedron,
            },
        },
        solid::elastic::ElasticFiniteElementModel,
    },
    math::TensorVec,
};

#[test]
fn f3d() {
    let constitutive = NeoHookean {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    };
    let hexahedra = Block::<_, Hexahedron, 20>::new(
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    );
    let pyramids = Block::<_, Pyramid, 13>::new(
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    );
    let tetrahedra = Block::<_, Tetrahedron, 10>::new(
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    );
    let wedges = Block::<_, Wedge, 15>::new(
        constitutive.clone(),
        vec![],
        NodalReferenceCoordinates::new(),
    );
    let blocks = wedges;
    // let blocks = Blocks(pyramids, wedges);
    // let blocks = Blocks(hexahedra, (pyramids, (tetrahedra, wedges)));
    let model = Model {
        blocks,
        coordinates: NodalReferenceCoordinates::new(),
    };
    model.nodal_forces(&NodalCoordinates::new()).unwrap();
    todo!()
}
