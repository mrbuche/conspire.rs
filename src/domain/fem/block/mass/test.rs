use crate::{
    fem::{
        Blocks, Model, NodalReferenceCoordinates,
        block::{
            Block, ElementDensities,
            element::{FiniteElement, linear::Tetrahedron},
        },
    },
    math::{Quantity, Tensor},
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
};

const EPSILON: f64 = 1e-12;

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
            fn conserves_the_mass_of_the_block() {
                let block = block();
                let mass = block.mass();
                let model = Model::from((block, coordinates()));
                let lumped = model.nodal_lumped_masses();
                assert!(!total(lumped.iter().copied()).differs(mass, EPSILON));
            }

            #[test]
            fn shared_nodes_collect_from_every_element() {
                let reference = |nodes: [usize; 4]| {
                    let coordinates = coordinates();
                    Tetrahedron::<$g>::from(
                        nodes
                            .iter()
                            .map(|&node| coordinates[node].clone())
                            .collect::<crate::fem::block::element::ElementNodalReferenceCoordinates<4>>(),
                    )
                    .volume()
                };
                let first = DENSITY * reference(CONNECTIVITY[0]) / 4.0;
                let second = DENSITY * reference(CONNECTIVITY[1]) / 4.0;
                let model = Model::from((block(), coordinates()));
                let lumped = model.nodal_lumped_masses();
                assert!(!lumped[0].differs(first, EPSILON));
                assert!(!lumped[4].differs(second, EPSILON));
                (1..4).for_each(|node| assert!(!lumped[node].differs(first + second, EPSILON)));
            }

            #[test]
            fn follows_a_density_that_varies() {
                let block = Block::<(), Tetrahedron<$g>, $g, 3, 4, 4, ElementDensities<$g>>::from((
                    (),
                    |coordinate: &ReferenceCoordinate| {
                        Density::kilograms_per_cubic_meter(1e3 + 5e2 * coordinate[0].value())
                    },
                    CONNECTIVITY.to_vec(),
                    &coordinates(),
                ));
                let mass = block.mass();
                let model = Model::from((block, coordinates()));
                assert!(!total(model.nodal_lumped_masses().iter().copied()).differs(mass, EPSILON));
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
    fn conserves_the_mass_of_the_block() {
        let model = model();
        let mass = DENSITY * model.blocks.volume();
        let masses = model.nodal_masses();
        let sum = total(
            masses
                .iter()
                .flat_map(|row| row.entries().map(|(_, entry)| *entry)),
        );
        assert!(!sum.differs(mass, EPSILON));
    }

    #[test]
    fn is_symmetric() {
        let masses = model().nodal_masses();
        masses.iter().enumerate().for_each(|(a, row)| {
            row.entries()
                .for_each(|(b, entry)| assert!(!masses[b][a].differs(*entry, EPSILON)))
        });
    }

    #[test]
    fn couples_only_nodes_that_share_an_element() {
        let masses = model().nodal_masses();
        assert!(masses[0].entries().all(|(b, _)| b != 4));
        assert!(masses[4].entries().all(|(b, _)| b != 0));
    }

    #[test]
    fn lumps_to_its_row_sums() {
        let model = model();
        let masses = model.nodal_masses();
        let lumped = model.nodal_lumped_masses();
        masses.iter().zip(lumped.iter()).for_each(|(row, lumped)| {
            let sum = total(row.entries().map(|(_, entry)| *entry));
            assert!(!sum.differs(*lumped, EPSILON))
        });
    }
}

mod combined {
    use super::*;

    type B = Block<(), Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;

    #[test]
    fn blocks_add_their_masses() {
        let heavy = Density::kilograms_per_cubic_meter(2.0 * 7.8e3);
        let first = B::from(((), DENSITY, vec![CONNECTIVITY[0]], &coordinates()));
        let second = B::from(((), heavy, vec![CONNECTIVITY[1]], &coordinates()));
        let mass = first.mass() + second.mass();
        let model = Model::from((Blocks(first, second), coordinates()));
        assert!(!total(model.nodal_lumped_masses().iter().copied()).differs(mass, EPSILON));
        let consistent = total(
            model
                .nodal_masses()
                .iter()
                .flat_map(|row| row.entries().map(|(_, entry)| *entry)),
        );
        assert!(!consistent.differs(mass, EPSILON));
    }
}
