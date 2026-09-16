use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::{
        Blocks, Model, NodalReferenceCoordinates, nodal_coordinates,
        solid::elastic::ElasticElements,
    },
    fem::block::{Block as FemBlock, element::linear::Tetrahedron},
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, PolytopalConnectivity},
    },
    math::Tensor,
    units::Stress,
    vem::block::{Block as VemBlock, element::Element as VemElement},
};

fn constitutive_model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn reference_coordinates() -> NodalReferenceCoordinates<3> {
    nodal_coordinates(Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ]))
}

type MixedBlocks =
    Blocks<FemBlock<NeoHookean, Tetrahedron, 1, 3, 4, 4>, VemBlock<NeoHookean, VemElement>>;

#[test]
fn fem_and_vem_blocks_combine_via_trait_bounds() {
    let coordinates = reference_coordinates();
    let fem_block = FemBlock::<NeoHookean, Tetrahedron, 1, 3, 4, 4>::from((
        constitutive_model(),
        vec![[8, 9, 10, 11]],
        &coordinates,
    ));
    let vem_block = VemBlock::<NeoHookean, VemElement>::from((
        constitutive_model(),
        vec![vec![0, 1, 2, 3, 4, 5]],
        vec![
            vec![0, 2, 3, 1],
            vec![4, 5, 7, 6],
            vec![0, 1, 5, 4],
            vec![2, 6, 7, 3],
            vec![0, 4, 6, 2],
            vec![1, 3, 7, 5],
        ],
        &coordinates,
    ));
    let model: Model<MixedBlocks, 3> = (Blocks(fem_block, vem_block), coordinates.clone()).into();
    let current_coordinates = coordinates.into();
    let forces = ElasticElements::nodal_forces(&model, &current_coordinates)
        .expect("mixed fem/vem model should compute nodal forces");
    assert_eq!(forces.len(), 12);
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

type ThreeBlocks = Blocks<
    Blocks<
        FemBlock<NeoHookean, Tetrahedron, 1, 3, 4, 4>,
        FemBlock<NeoHookean, Tetrahedron, 1, 3, 4, 4>,
    >,
    VemBlock<NeoHookean, VemElement>,
>;

#[test]
fn three_blocks_two_fem_one_vem_combine_via_mesh() {
    let mesh = Mesh::from((
        vec![
            Connectivity::Tetrahedral(vec![[8, 9, 10, 11]].into()),
            Connectivity::Tetrahedral(vec![[8, 9, 10, 11]].into()),
            Connectivity::Polyhedral(PolytopalConnectivity::from((
                elements_faces(),
                faces_nodes(),
            ))),
        ],
        Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]),
    ));
    let model: Model<ThreeBlocks, 3> = (
        mesh,
        (
            (constitutive_model(), constitutive_model()),
            constitutive_model(),
        ),
    )
        .try_into()
        .expect("mesh with 3 connectivity groups should build a 3-block model");
    let current_coordinates = model.coordinates.clone().into();
    let forces = ElasticElements::nodal_forces(&model, &current_coordinates)
        .expect("3-block mixed model should compute nodal forces");
    assert_eq!(forces.len(), 12);
}

#[test]
fn element_error_messages_name_their_kind() {
    use crate::fem::block::element::FiniteElementError;
    use crate::vem::block::element::VirtualElementError;
    let fem_message = FiniteElementError::upstream("bad model", &"ctx").to_string();
    let vem_message = VirtualElementError::upstream("bad model", &"ctx").to_string();
    assert!(fem_message.contains("In finite element: \"ctx\"."));
    assert!(vem_message.contains("In virtual element: \"ctx\"."));
}
