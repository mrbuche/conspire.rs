use crate::{
    fem::Model,
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh, PolytopalConnectivity},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
    vem::{
        NodalReferenceCoordinates,
        block::{Block, element::Element},
    },
};

type VemBlock<R = crate::vem::block::NoDensity> = Block<(), Element, R>;

const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

fn problem() -> (PolytopalConnectivity<3>, NodalReferenceCoordinates) {
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) =
        Mesh::from_voxels(Voxels::new(vec![1u8; 2], [2, 1, 1]), None).into();
    let hexahedra: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|connectivity| match connectivity {
            Connectivity::Hexahedral(hexahedra) => hexahedra.into_iter().collect::<Vec<_>>(),
            _ => panic!("expected a hexahedral mesh"),
        })
        .collect();
    let faces_nodes: Vec<Vec<usize>> = hexahedra
        .iter()
        .flat_map(|hexahedron| {
            FACES
                .iter()
                .map(|face| face.iter().map(|&node| hexahedron[node]).collect())
        })
        .collect();
    let elements_faces: Vec<Vec<usize>> = (0..hexahedra.len())
        .map(|element| (6 * element..6 * element + 6).collect())
        .collect();
    (
        (elements_faces, faces_nodes).into(),
        coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    )
}

const LEFT: Quantity<Density> = Density::kilograms_per_cubic_meter(1e3);
const RIGHT: Quantity<Density> = Density::kilograms_per_cubic_meter(3e3);

fn field(coordinate: &ReferenceCoordinate) -> Quantity<Density> {
    if coordinate[0].value() < 1.0 {
        LEFT
    } else {
        RIGHT
    }
}

#[test]
fn a_uniform_density_block_has_the_mass_of_its_volume() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = VemBlock::from(((), LEFT, connectivity, &coordinates));
    let model = Model::from((block, coordinates));
    let masses = model.nodal_lumped_masses();
    let total: Quantity<Mass> = masses.iter().copied().sum();
    Assert::default().eq_within_tols(total, &(LEFT * crate::units::Volume::cubic_meters(2.0)))
}

#[test]
fn the_nodal_masses_are_an_eighth_of_each_hexahedron_mass() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = VemBlock::from(((), field, connectivity, &coordinates));
    let model = Model::from((block, coordinates));
    let masses = model.nodal_lumped_masses();
    let left = LEFT * crate::units::Volume::cubic_meters(1.0) / 8.0;
    let right = RIGHT * crate::units::Volume::cubic_meters(1.0) / 8.0;
    let (connectivity, _) = problem();
    let mut expected = vec![left * 0.0; masses.len()];
    connectivity
        .iter()
        .enumerate()
        .for_each(|(element, faces)| {
            let share = if element == 0 { left } else { right };
            let mut nodes: Vec<usize> = faces
                .iter()
                .flat_map(|&face| connectivity.faces_nodes()[face].clone())
                .collect();
            nodes.sort();
            nodes.dedup();
            nodes.iter().for_each(|&node| expected[node] += share)
        });
    masses
        .iter()
        .zip(&expected)
        .try_for_each(|(mass, expected)| Assert::default().eq_within_tols(mass, expected))
}

#[test]
fn the_block_mass_is_the_total_of_the_nodal_masses() -> Result<(), AssertionError> {
    let (connectivity, coordinates) = problem();
    let block = VemBlock::from(((), field, connectivity, &coordinates));
    let mass = block.mass();
    let model = Model::from((block, coordinates));
    let total: Quantity<Mass> = model.nodal_lumped_masses().iter().copied().sum();
    Assert::default().eq_within_tols(total, &mass)
}
