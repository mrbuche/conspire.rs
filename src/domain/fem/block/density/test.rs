use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, ElementDensities, element::linear::Tetrahedron},
    },
    mechanics::ReferenceCoordinate,
    units::Density,
};

const EPSILON: f64 = 1e-12;

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

            #[test]
            fn constant_density_gives_density_times_volume() {
                let density = Density::kilograms_per_cubic_meter(7.8e3);
                let block = B::from(((), density, CONNECTIVITY.to_vec(), &coordinates()));
                assert!(!block.mass().differs(density * block.volume(), EPSILON));
                assert!(
                    block
                        .density()
                        .iter()
                        .flatten()
                        .all(|point_density| !point_density.differs(density, EPSILON))
                );
            }

            #[test]
            fn variable_density_is_sampled_at_the_integration_points() {
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
                    .sum();
                assert!(!block.mass().differs(expected, EPSILON));
            }
        }
    };
}

test_density!(one_point, 1);
test_density!(four_points, 4);
