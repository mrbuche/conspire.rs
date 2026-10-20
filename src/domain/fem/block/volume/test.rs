use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Blocks, Model, NodalReferenceCoordinates,
        block::{Block, element::linear::Hexahedron},
    },
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::{Stress, Volume},
};

type Hex = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;

fn neo_hookean() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    }
}

fn coordinates() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 3.0, 1.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 3.0, 2.0],
        [0.0, 3.0, 2.0],
    ])
}

fn stack() -> Model<Hex, 3> {
    let coordinates = coordinates();
    (
        Hex::from((
            neo_hookean(),
            vec![[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]],
            &coordinates,
        )),
        coordinates,
    )
        .into()
}

fn near(found: Quantity<Volume>, expected: f64) -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(found, &Quantity::<Volume>::new(expected))
}

#[test]
fn a_cuboid_gives_an_eighth_of_its_volume_to_each_node() -> Result<(), AssertionError> {
    let coordinates = coordinates();
    let model: Model<Hex, 3> = (
        Hex::from((neo_hookean(), vec![[0, 1, 2, 3, 4, 5, 6, 7]], &coordinates)),
        coordinates,
    )
        .into();
    let volumes = model.nodal_volumes();
    (0..8).try_for_each(|node| near(volumes[node], 6.0 / 8.0))
}

#[test]
fn a_node_shared_by_elements_has_the_volume_of_each() -> Result<(), AssertionError> {
    let volumes = stack().nodal_volumes();
    (0..4).try_for_each(|node| near(volumes[node], 0.75))?;
    (4..8).try_for_each(|node| near(volumes[node], 1.5))?;
    (8..12).try_for_each(|node| near(volumes[node], 0.75))
}

#[test]
fn the_nodal_volumes_add_up_to_the_volume() -> Result<(), AssertionError> {
    near(stack().nodal_volumes().volume(), 12.0)
}

#[test]
fn blocks_add_up_too() -> Result<(), AssertionError> {
    let coordinates = coordinates();
    let model: Model<Blocks<Hex, Hex>, 3> = (
        Blocks(
            Hex::from((neo_hookean(), vec![[0, 1, 2, 3, 4, 5, 6, 7]], &coordinates)),
            Hex::from((
                neo_hookean(),
                vec![[4, 5, 6, 7, 8, 9, 10, 11]],
                &coordinates,
            )),
        ),
        coordinates,
    )
        .into();
    near(model.nodal_volumes().volume(), 12.0)
}
