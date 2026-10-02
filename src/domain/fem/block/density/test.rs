use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, Densities, ElementDensities, element::linear::Tetrahedron},
    },
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
};

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

fn linear(x: f64) -> f64 {
    1e3 + 5e2 * x
}

fn centroid_x(nodes: [usize; 4]) -> f64 {
    let coordinates = coordinates();
    nodes
        .iter()
        .map(|&node| coordinates[node][0].value())
        .sum::<f64>()
        / 4.0
}

macro_rules! test_density {
    ($mod:ident, $g:literal) => {
        mod $mod {
            use super::*;

            type B = Block<(), Tetrahedron<$g>, $g, 3, 4, 4, ElementDensities<$g>>;
            type Scalar = Block<(), Tetrahedron<$g>, $g, 3, 4, 4, Quantity<Density>>;

            #[test]
            fn constant_density_gives_density_times_volume() -> Result<(), AssertionError> {
                let density = Density::kilograms_per_cubic_meter(7.8e3);
                let block = Scalar::from(((), density, CONNECTIVITY.to_vec(), &coordinates()));
                Assert::default().eq_within_tols(&block.mass(), &(density * block.volume()))?;
                Assert::default().eq_within_tols(block.density(), &density)
            }

            #[test]
            fn a_constant_closure_agrees_with_the_scalar() -> Result<(), AssertionError> {
                let density = Density::kilograms_per_cubic_meter(7.8e3);
                let scalar = Scalar::from(((), density, CONNECTIVITY.to_vec(), &coordinates()));
                let per_point = B::from((
                    (),
                    |_: &ReferenceCoordinate| density,
                    CONNECTIVITY.to_vec(),
                    &coordinates(),
                ));
                Assert::default().eq_within_tols(&per_point.mass(), &scalar.mass())?;
                (0..CONNECTIVITY.len()).try_for_each(|element| {
                    Assert::default().eq_within_tols(
                        &per_point.density().at(element),
                        &scalar.density().at(element),
                    )
                })
            }

            #[test]
            fn variable_density_is_sampled_at_the_integration_points() -> Result<(), AssertionError>
            {
                let block = B::from((
                    (),
                    |coordinate: &ReferenceCoordinate| {
                        Density::kilograms_per_cubic_meter(linear(coordinate[0].value()))
                    },
                    CONNECTIVITY.to_vec(),
                    &coordinates(),
                ));
                let reference = Block::<(), Tetrahedron<$g>, $g, 3, 4, 4>::from((
                    (),
                    CONNECTIVITY.to_vec(),
                    &coordinates(),
                ));
                let expected = CONNECTIVITY
                    .iter()
                    .zip(reference.elements())
                    .map(|(&nodes, element)| {
                        Density::kilograms_per_cubic_meter(linear(centroid_x(nodes)))
                            * crate::fem::block::element::FiniteElement::volume(element)
                    })
                    .sum::<Quantity<Mass>>();
                Assert::default().eq_within_tols(&block.mass(), &expected)
            }
        }
    };
}

test_density!(one_point, 1);
test_density!(four_points, 4);
