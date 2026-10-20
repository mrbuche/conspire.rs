use crate::{
    math::{Quantity, Scalar, Tensor},
    mechanics::ReferenceCoordinate,
    units::{Density, Mass, SpecificHeat, VolumetricHeatCapacity},
    vem::{
        NodalReferenceCoordinates,
        block::element::{
            DEFAULT_STABILIZATION, Element, ElementNodalReferenceCoordinates, VirtualElement,
            mass::LumpedMassVirtualElement, thermal::capacity::LumpedHeatCapacityVirtualElement,
        },
    },
};

type Case = (Element, NodalReferenceCoordinates);

fn capacity() -> Quantity<VolumetricHeatCapacity> {
    Density::kilograms_per_cubic_meter(7.8e3) * SpecificHeat::joules_per_kilogram_kelvin(450.0)
}

fn element(nodes: &[[Scalar; 3]], faces: &[Vec<usize>]) -> Case {
    let coordinates: ElementNodalReferenceCoordinates = faces
        .iter()
        .map(|face| {
            face.iter()
                .map(|&node| ReferenceCoordinate::from(nodes[node]))
                .collect()
        })
        .collect();
    let faces_indices = (0..faces.len()).collect::<Vec<_>>();
    let nodes_indices = (0..nodes.len()).collect::<Vec<_>>();
    let nodal_coordinates = NodalReferenceCoordinates::from(
        nodes
            .iter()
            .map(|&node| ReferenceCoordinate::from(node))
            .collect::<Vec<_>>(),
    );
    (
        (
            coordinates,
            &faces_indices[..],
            &nodes_indices[..],
            faces,
            DEFAULT_STABILIZATION,
        )
            .into(),
        nodal_coordinates,
    )
}

fn cases() -> Vec<Case> {
    let s = 3.0_f64.sqrt() / 2.0;
    vec![
        element(
            &[
                [0.1, 0.2, 0.0],
                [1.3, 0.1, 0.2],
                [0.2, 0.9, 0.1],
                [0.3, 0.4, 1.2],
            ],
            &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        ),
        element(
            &[
                [0.0, 0.0, 0.0],
                [2.0, 0.1, 0.0],
                [2.2, 1.0, 0.1],
                [0.0, 1.0, 0.0],
                [0.1, 0.0, 3.0],
                [2.0, 0.0, 3.3],
                [2.1, 1.2, 3.0],
                [0.0, 1.0, 2.8],
            ],
            &[
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![3, 7, 6, 2],
                vec![0, 4, 7, 3],
                vec![1, 2, 6, 5],
            ],
        ),
        element(
            &[
                [1.0, 0.0, 0.0],
                [-0.5, s, 0.0],
                [-0.5, -s, 0.0],
                [0.0, 0.1, 0.7],
                [0.1, 0.0, -0.4],
            ],
            &[
                vec![0, 1, 3],
                vec![1, 2, 3],
                vec![2, 0, 3],
                vec![1, 0, 4],
                vec![2, 1, 4],
                vec![0, 2, 4],
            ],
        ),
    ]
}

#[test]
fn totals_the_heat_capacity_of_the_element() {
    cases().iter().for_each(|(element, coordinates)| {
        let total = element
            .nodal_lumped_heat_capacities(capacity(), coordinates)
            .iter()
            .copied()
            .sum::<Quantity<_>>();
        let expected = capacity() * element.volume();
        assert!(
            ((total / expected).value() - 1.0).abs() < 1e-12,
            "{total:?} != {expected:?}"
        )
    })
}

#[test]
fn lumps_like_the_mass() {
    let density = Density::kilograms_per_cubic_meter(7.8e3);
    cases().iter().for_each(|(element, coordinates)| {
        let masses = element.nodal_lumped_masses(density, coordinates);
        let capacities = element.nodal_lumped_heat_capacities(capacity(), coordinates);
        let total_mass = masses.iter().copied().sum::<Quantity<Mass>>();
        let total_capacity = capacities.iter().copied().sum::<Quantity<_>>();
        masses
            .iter()
            .zip(capacities.iter())
            .for_each(|(&mass, &capacity)| {
                assert!(
                    ((mass / total_mass).value() - (capacity / total_capacity).value()).abs()
                        < 1e-12
                )
            })
    })
}
