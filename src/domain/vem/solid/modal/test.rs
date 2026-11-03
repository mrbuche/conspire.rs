use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::modal::test::{ELEMENTS, coordinates, density, lumped_bar_tests, material},
    math::Quantity,
    units::Density,
    vem::{
        Model,
        block::{
            Block,
            element::{DEFAULT_STABILIZATION, Element},
        },
    },
};

type Cells = Block<NeoHookean, Element, Quantity<Density>>;

const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

fn cells() -> Model<Cells, 3> {
    let reference = coordinates().into();
    let faces_nodes: Vec<Vec<usize>> = (0..ELEMENTS)
        .flat_map(|k| {
            FACES
                .iter()
                .map(move |face| face.iter().map(|&node| 4 * k + node).collect())
        })
        .collect();
    let elements_faces: Vec<Vec<usize>> = (0..ELEMENTS)
        .map(|k| (6 * k..6 * k + 6).collect())
        .collect();
    (
        Block::<_, Element>::from((
            material(),
            elements_faces,
            faces_nodes,
            &reference,
            DEFAULT_STABILIZATION,
        ))
        .with_density(density()),
        reference,
    )
        .into()
}

mod hexahedra {
    use super::*;
    lumped_bar_tests!(cells, 1e-8, below);
}
