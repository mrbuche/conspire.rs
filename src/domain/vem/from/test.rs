use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{Model, solid::elastic::ElasticElements},
    geometry::mesh::{Connectivity, Mesh, PolytopalConnectivity},
    math::{
        Tensor,
        test::{TestError, assert_eq},
    },
    vem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::Element},
    },
};

fn constitutive_model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    }
}

fn coordinates() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
}

fn deformed_coordinates() -> NodalCoordinates {
    coordinates()
        .iter()
        .map(|coordinate| {
            [
                1.1 * coordinate[0] + 0.05 * coordinate[1],
                0.95 * coordinate[1],
                1.05 * coordinate[2],
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn elements_faces() -> Vec<Vec<usize>> {
    vec![vec![0, 1, 2, 3, 4, 5]]
}

fn faces_nodes() -> Vec<Vec<usize>> {
    vec![
        vec![0, 2, 3, 1],
        vec![4, 5, 7, 6],
        vec![0, 1, 5, 4],
        vec![2, 6, 7, 3],
        vec![0, 4, 6, 2],
        vec![1, 3, 7, 5],
    ]
}

#[test]
fn polyhedral_block_nodal_forces() -> Result<(), TestError> {
    let mesh = Mesh::from((
        vec![Connectivity::Polyhedral(PolytopalConnectivity::from((
            elements_faces(),
            faces_nodes(),
        )))],
        coordinates(),
    ));
    let model: Model<Block<NeoHookean, Element>, 3> = (mesh, constitutive_model())
        .try_into()
        .map_err(|error: String| TestError { message: error })?;
    let block = Block::<NeoHookean, Element>::from((
        constitutive_model(),
        elements_faces(),
        faces_nodes(),
        &coordinates(),
    ));
    assert_eq(
        &block.nodal_forces(&deformed_coordinates())?,
        &ElasticElements::nodal_forces(&model, &deformed_coordinates())?,
    )
}

#[test]
fn wrong_element_kind() {
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(
            vec![[0, 1, 2, 4], [1, 3, 2, 7]].into(),
        )],
        coordinates(),
    ));
    let model: Result<Model<Block<NeoHookean, Element>, 3>, String> =
        (mesh, constitutive_model()).try_into();
    assert_eq!(model.err().unwrap(), "block is not polyhedral")
}
