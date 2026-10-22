use super::{CohesiveBlock, DamageStateField};
use crate::{
    constitutive::{
        cohesive::{damage::TvergaardHutchinson, elastic::LinearElastic},
        solid::hyperelastic::NeoHookean,
    },
    fem::{
        Blocks, Model, NodalCoordinates,
        block::{
            Block,
            element::{
                ElementNodalReferenceCoordinates,
                cohesive::{
                    damage::{DamageCohesiveElement, initial_state},
                    linear::Hexahedron as CohesiveHexahedron,
                },
                linear::Hexahedron,
            },
        },
        solid::NodalStiffnessesSolid,
    },
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::{
        Quantity, Scalar, Tensor,
        assert::{Assert, AssertionError},
    },
    units::{Force, Length, Stress, StressPerLength},
};
use std::array::from_fn;

type Damaged = CohesiveBlock<TvergaardHutchinson, CohesiveHexahedron, 4, 8, 4>;
type Elastic = CohesiveBlock<LinearElastic, CohesiveHexahedron, 4, 8, 4>;
type Solid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;

const ELEMENTS: [[usize; 8]; 2] = [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]];
const BOTTOM: [[Scalar; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [2.0, 1.0, 0.0],
];

fn reference() -> [[Scalar; 3]; 12] {
    from_fn(|node| BOTTOM[node % 6])
}

fn coordinates() -> Coordinates<3> {
    reference().into()
}

fn damage_model() -> TvergaardHutchinson {
    TvergaardHutchinson {
        strength: Stress::pascals(2.0),
        normal_length: Length::meters(0.1),
        tangential_length: Length::meters(0.2),
        onset: 0.2,
        softening: 0.6,
    }
}

fn elastic_model() -> LinearElastic {
    LinearElastic {
        normal_stiffness: StressPerLength::pascals_per_meter(3.4),
        tangential_stiffness: StressPerLength::pascals_per_meter(5.6),
    }
}

fn mesh() -> Mesh<3> {
    (
        vec![Connectivity::Hexahedral(ELEMENTS.to_vec().into())],
        coordinates(),
    )
        .into()
}

fn opened(opening: Scalar) -> NodalCoordinates<3> {
    let mut coordinates = reference();
    coordinates
        .iter_mut()
        .skip(6)
        .for_each(|coordinate| coordinate[2] += opening);
    coordinates.into()
}

fn damage() -> Result<Model<Damaged, 3>, AssertionError> {
    (mesh(), damage_model())
        .try_into()
        .map_err(|message: String| AssertionError { message })
}

fn force(opening: Scalar, node: usize) -> Result<Quantity<Force>, AssertionError> {
    use crate::fem::cohesive::DamageElements;
    let model = damage()?;
    let state = model.initial_state();
    Ok(model.nodal_forces(&opened(opening), &state)?[node][2])
}

#[test]
fn the_forces_of_the_elements_add_at_the_shared_nodes() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(force(0.05, 6)?, &Quantity::new(0.5))?;
    Assert::default().eq_within_tols(force(0.05, 7)?, &Quantity::new(1.0))?;
    Assert::default().eq_within_tols(force(0.05, 10)?, &Quantity::new(1.0))?;
    Assert::default().eq_within_tols(force(0.05, 1)?, &Quantity::new(-1.0))?;
    Assert::default().eq_within_tols(force(0.0, 6)?, &Quantity::new(0.0))
}

#[test]
fn the_stiffnesses_of_the_elements_add_at_the_shared_nodes() -> Result<(), AssertionError> {
    use crate::fem::cohesive::DamageElements;
    let model = damage()?;
    let state = model.initial_state();
    let coordinates = opened(0.05);
    let stiffnesses = model.nodal_stiffnesses(&coordinates, &state)?;
    let mut expected = NodalStiffnessesSolid::<3>::zero(coordinates.len());
    ELEMENTS.iter().try_for_each(|nodes| {
        let element = CohesiveHexahedron::from(ElementNodalReferenceCoordinates::from(
            std::array::from_fn::<_, 8, _>(|local| reference()[nodes[local]]),
        ));
        let element_coordinates = nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect();
        let element_stiffnesses = DamageCohesiveElement::nodal_stiffnesses(
            &element,
            &damage_model(),
            &element_coordinates,
            &initial_state::<4>(),
        )?;
        element_stiffnesses
            .into_iter()
            .zip(nodes)
            .for_each(|(object, &a)| {
                object
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(stiffness, &b)| expected[a][b] += stiffness)
            });
        Ok::<(), AssertionError>(())
    })?;
    ELEMENTS.iter().flatten().try_for_each(|&a| {
        ELEMENTS
            .iter()
            .filter(|nodes| nodes.contains(&a))
            .flatten()
            .try_for_each(|&b| {
                Assert::default().eq_within_tols(&stiffnesses[a][b], &expected[a][b])
            })
    })
}

#[test]
fn the_state_is_kept_by_element_and_the_step_error_is_the_largest() -> Result<(), AssertionError> {
    use crate::fem::cohesive::DamageElements;
    let model = damage()?;
    let state = model.initial_state();
    assert_eq!(state.len(), 2);
    let coordinates = opened(0.05);
    let updated: DamageStateField<4> = model.updated_state(&coordinates, &state)?;
    let uniform = |opening: Scalar| -> DamageStateField<4> {
        (0..2)
            .map(|_| (0..4).map(|_| Quantity::new(opening)).collect())
            .collect()
    };
    for (element, expected) in updated.iter().zip(uniform(0.5)) {
        Assert::default().eq_within_tols(element, &expected)?;
    }
    let error = model.step_error(&opened(0.0), &coordinates, &state, &updated)?;
    Assert::default().eq_within_tols(error.direction.value(), 0.0)?;
    Assert::default().eq_within_tols(error.overshoot.value(), 0.3)?;
    let mut uneven = opened(0.05);
    uneven[11][2] += Quantity::new(0.12);
    let updated_uneven = model.updated_state(&uneven, &state)?;
    assert_eq!(updated_uneven[0], updated[0]);
    assert!(
        updated_uneven[1]
            .iter()
            .any(|opening| opening.value() > 0.5)
    );
    let error = model.step_error(&opened(0.0), &uneven, &state, &updated_uneven)?;
    assert!(error.overshoot.value() > 0.3);
    Ok(())
}

#[test]
fn the_elastic_block_gives_the_forces_of_its_elements() -> Result<(), AssertionError> {
    use crate::fem::solid::elastic::ElasticElements;
    let model: Model<Elastic, 3> = (mesh(), elastic_model())
        .try_into()
        .map_err(|message: String| AssertionError { message })?;
    let forces = model.nodal_forces(&opened(0.05))?;
    Assert::default().eq_within_tols(forces[6][2], &Quantity::<Force>::new(0.5 * 0.17))?;
    Assert::default().eq_within_tols(forces[7][2], &Quantity::<Force>::new(0.17))?;
    Assert::default().eq_within_tols(forces[0][2], &Quantity::<Force>::new(-0.5 * 0.17))
}

#[test]
fn a_cohesive_block_composes_with_a_solid_block() -> Result<(), AssertionError> {
    use crate::fem::solid::elastic::ElasticElements;
    let mut positions = [[0.0; 3]; 12];
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .iter()
    .zip(positions.iter_mut())
    .for_each(|(position, slot)| *slot = *position);
    let mesh = Mesh::from((
        vec![
            Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
            Connectivity::Hexahedral(vec![[4, 5, 6, 7, 8, 9, 10, 11]].into()),
        ],
        Coordinates::from(positions),
    ));
    let solid = NeoHookean {
        bulk_modulus: Stress::pascals(13.0),
        shear_modulus: Stress::pascals(3.0),
    };
    let model: Model<Blocks<Solid, Elastic>, 3> = (mesh, (solid, elastic_model()))
        .try_into()
        .map_err(|message: String| AssertionError { message })?;
    let mut coordinates = positions;
    coordinates
        .iter_mut()
        .skip(8)
        .for_each(|coordinate| coordinate[2] += 0.05);
    let forces = model.nodal_forces(&NodalCoordinates::from(coordinates))?;
    (0..4).try_for_each(|node| {
        Assert::default().eq_within_tols(forces[node][2], &Quantity::<Force>::new(0.0))?;
        Assert::default()
            .eq_within_tols(forces[4 + node][2], &Quantity::<Force>::new(-0.17 / 4.0))?;
        Assert::default().eq_within_tols(forces[8 + node][2], &Quantity::<Force>::new(0.17 / 4.0))
    })
}

#[test]
fn a_mesh_with_two_blocks_is_not_a_cohesive_block() {
    let mesh = Mesh::from((
        vec![
            Connectivity::Hexahedral(vec![ELEMENTS[0]].into()),
            Connectivity::Hexahedral(vec![ELEMENTS[1]].into()),
        ],
        coordinates(),
    ));
    let model: Result<Model<Damaged, 3>, String> = (mesh, damage_model()).try_into();
    assert!(model.is_err());
}
