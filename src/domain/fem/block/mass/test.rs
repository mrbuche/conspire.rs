use crate::{
    fem::{
        Blocks, Model, NodalReferenceCoordinates,
        block::{
            Block, ElementDensities,
            element::{ElementNodalReferenceCoordinates, FiniteElement, linear::Tetrahedron},
            mass::NodalLumpedMasses,
        },
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn coordinates() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ])
}

const CONNECTIVITY: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 3, 4]];

fn total(masses: impl Iterator<Item = Quantity<Mass>>) -> Quantity<Mass> {
    masses.sum()
}

macro_rules! test_lumped {
    ($mod:ident, $g:literal) => {
        mod $mod {
            use super::*;
            type B = Block<(), Tetrahedron<$g>, $g, 3, 4, 4, Quantity<Density>>;
            fn block() -> B {
                B::from(((), DENSITY, CONNECTIVITY.to_vec(), &coordinates()))
            }
            #[test]
            fn conserves_the_mass_of_the_block() -> Result<(), AssertionError> {
                let block = block();
                let mass = block.mass();
                let model = Model::from((block, coordinates()));
                Assert::default()
                    .eq_within_tols(&total(model.nodal_lumped_masses().iter().copied()), &mass)
            }
            #[test]
            fn shared_nodes_collect_from_every_element() -> Result<(), AssertionError> {
                let volume = |nodes: [usize; 4]| {
                    let coordinates = coordinates();
                    Tetrahedron::<$g>::from(
                        nodes
                            .iter()
                            .map(|&node| coordinates[node].clone())
                            .collect::<ElementNodalReferenceCoordinates<4>>(),
                    )
                    .volume()
                };
                let first = DENSITY * volume(CONNECTIVITY[0]) / 4.0;
                let second = DENSITY * volume(CONNECTIVITY[1]) / 4.0;
                let expected = NodalLumpedMasses::from([
                    first,
                    first + second,
                    first + second,
                    first + second,
                    second,
                ]);
                Assert::default().eq_within_tols(
                    &Model::from((block(), coordinates())).nodal_lumped_masses(),
                    &expected,
                )
            }
            #[test]
            fn follows_a_density_that_varies() -> Result<(), AssertionError> {
                let block =
                    Block::<(), Tetrahedron<$g>, $g, 3, 4, 4, ElementDensities<$g>>::from((
                        (),
                        |coordinate: &ReferenceCoordinate| {
                            Density::kilograms_per_cubic_meter(1e3 + 5e2 * coordinate[0].value())
                        },
                        CONNECTIVITY.to_vec(),
                        &coordinates(),
                    ));
                let mass = block.mass();
                let model = Model::from((block, coordinates()));
                Assert::default()
                    .eq_within_tols(&total(model.nodal_lumped_masses().iter().copied()), &mass)
            }
        }
    };
}

test_lumped!(lumped_one_point, 1);
test_lumped!(lumped_four_points, 4);

mod consistent {
    use super::*;
    type B = Block<(), Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;
    fn model() -> Model<B, 3> {
        Model::from((
            B::from(((), DENSITY, CONNECTIVITY.to_vec(), &coordinates())),
            coordinates(),
        ))
    }
    #[test]
    fn conserves_the_mass_of_the_block() -> Result<(), AssertionError> {
        let model = model();
        let mass = DENSITY * model.blocks.volume();
        let sum = total(
            model
                .nodal_masses()
                .iter()
                .flat_map(|row| row.entries().map(|(_, entry)| *entry)),
        );
        Assert::default().eq_within_tols(sum, &mass)
    }
    #[test]
    fn is_symmetric() -> Result<(), AssertionError> {
        let masses = model().nodal_masses();
        masses.iter().enumerate().try_for_each(|(a, row)| {
            row.entries()
                .try_for_each(|(b, entry)| Assert::default().eq_within_tols(masses[b][a], entry))
        })
    }
    #[test]
    fn couples_only_nodes_that_share_an_element() {
        let masses = model().nodal_masses();
        assert!(masses[0].entries().all(|(b, _)| b != 4));
        assert!(masses[4].entries().all(|(b, _)| b != 0));
    }
    #[test]
    fn lumps_to_its_row_sums() -> Result<(), AssertionError> {
        let model = model();
        let row_sums = model
            .nodal_masses()
            .iter()
            .map(|row| total(row.entries().map(|(_, entry)| *entry)))
            .collect::<NodalLumpedMasses>();
        Assert::default().eq_within_tols(model.nodal_lumped_masses(), &row_sums)
    }
}

mod combined {
    use super::*;
    type B = Block<(), Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;
    #[test]
    fn blocks_add_their_masses() -> Result<(), AssertionError> {
        let heavy = Density::kilograms_per_cubic_meter(2.0 * 7.8e3);
        let first = B::from(((), DENSITY, vec![CONNECTIVITY[0]], &coordinates()));
        let second = B::from(((), heavy, vec![CONNECTIVITY[1]], &coordinates()));
        let mass = first.mass() + second.mass();
        let model = Model::from((Blocks(first, second), coordinates()));
        Assert::default()
            .eq_within_tols(total(model.nodal_lumped_masses().iter().copied()), &mass)?;
        let consistent = total(
            model
                .nodal_masses()
                .iter()
                .flat_map(|row| row.entries().map(|(_, entry)| *entry)),
        );
        Assert::default().eq_within_tols(consistent, &mass)
    }
}
