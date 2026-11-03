use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::{
            plastic::{Linear, PlasticFlow, VonMises},
            viscoplastic::ViscoplasticFlow,
        },
        solid::hyperelastic::NeoHookean,
    },
    domain::solid::modal::test::{
        ELEMENTS, coordinates, density, flowing, lumped_bar_tests, material, plastic_bar_tests,
        plasticity, viscoplastic_bar_tests,
    },
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
type Flowing = Block<Canonical<NeoHookean, ViscoplasticFlow>, Element, Quantity<Density>>;
type Plastic =
    Block<Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>, Element, Quantity<Density>>;

const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

fn connectivity() -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let faces_nodes = (0..ELEMENTS)
        .flat_map(|k| {
            FACES
                .iter()
                .map(move |face| face.iter().map(|&node| 4 * k + node).collect())
        })
        .collect();
    let elements_faces = (0..ELEMENTS)
        .map(|k| (6 * k..6 * k + 6).collect())
        .collect();
    (elements_faces, faces_nodes)
}

fn cells() -> Model<Cells, 3> {
    let reference = coordinates().into();
    let (elements_faces, faces_nodes) = connectivity();
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

fn flowing_cells() -> Model<Flowing, 3> {
    let reference = coordinates().into();
    let (elements_faces, faces_nodes) = connectivity();
    (
        Block::<_, Element>::from((
            flowing(),
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

fn plastic_cells() -> Model<Plastic, 3> {
    let reference = coordinates().into();
    let (elements_faces, faces_nodes) = connectivity();
    (
        Block::<_, Element>::from((
            plasticity(),
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

mod viscoplastic {
    use super::*;
    viscoplastic_bar_tests!(flowing_cells, 1, 1.1, 1e-8);
}

mod plastic {
    use super::*;
    plastic_bar_tests!(plastic_cells, 1, 1.1, 1e-8);
}
