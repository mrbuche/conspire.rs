use crate::{
    EPSILON,
    constitutive::thermal::conduction::Fourier,
    math::{Quantity, Scalar, Tensor},
    mechanics::ReferenceCoordinate,
    units::{PowerPerLengthTemperature, Temperature},
    vem::block::element::{
        Element, ElementNodalReferenceCoordinates, VirtualElement,
        diffusion::DiffusionVirtualElement,
        thermal::{ElementNodalTemperatures, ThermalVirtualElement},
    },
};

type Nodes = Vec<[Scalar; 3]>;

const MODEL: Fourier = Fourier {
    thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(2.5),
};

fn element(nodes: &[[Scalar; 3]], faces: &[Vec<usize>], stabilization: Scalar) -> Element {
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
    (
        coordinates,
        &faces_indices[..],
        &nodes_indices[..],
        faces,
        stabilization,
    )
        .into()
}

fn tetrahedron(stabilization: Scalar) -> (Element, Nodes) {
    let nodes = vec![
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
    ];
    let faces = [vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]];
    (element(&nodes, &faces, stabilization), nodes)
}

fn hexahedron_from(nodes: Nodes, stabilization: Scalar) -> (Element, Nodes) {
    let faces = [
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![3, 7, 6, 2],
        vec![0, 4, 7, 3],
        vec![1, 2, 6, 5],
    ];
    (element(&nodes, &faces, stabilization), nodes)
}

fn skewed(stabilization: Scalar) -> (Element, Nodes) {
    hexahedron_from(
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [2.2, 1.0, 0.1],
            [0.0, 1.0, 0.0],
            [0.1, 0.0, 3.0],
            [2.0, 0.0, 3.3],
            [2.1, 1.2, 3.0],
            [0.0, 1.0, 2.8],
        ],
        stabilization,
    )
}

fn box_(stabilization: Scalar) -> (Element, Nodes) {
    hexahedron_from(
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ],
        stabilization,
    )
}

fn bipyramid(stabilization: Scalar) -> (Element, Nodes) {
    let s = 3.0_f64.sqrt() / 2.0;
    let nodes = vec![
        [1.0, 0.0, 0.0],
        [-0.5, s, 0.0],
        [-0.5, -s, 0.0],
        [0.0, 0.1, 0.7],
        [0.1, 0.0, -0.4],
    ];
    let faces = [
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
        vec![1, 0, 4],
        vec![2, 1, 4],
        vec![0, 2, 4],
    ];
    (element(&nodes, &faces, stabilization), nodes)
}

fn cases(stabilization: Scalar) -> Vec<(Element, Nodes)> {
    vec![
        tetrahedron(stabilization),
        skewed(stabilization),
        bipyramid(stabilization),
    ]
}

fn temperatures(values: &[Scalar]) -> ElementNodalTemperatures {
    values
        .iter()
        .map(|&value| Quantity::<Temperature>::new(value))
        .collect()
}

fn pseudorandom(count: usize) -> Vec<Scalar> {
    (0..count)
        .map(|index| ((index as Scalar + 1.0) * 12.9898).sin() * 0.5 + 0.7)
        .collect()
}

fn linear(nodes: &[[Scalar; 3]], gradient: [Scalar; 3]) -> Vec<Scalar> {
    nodes
        .iter()
        .map(|node| {
            2.0 + (0..3)
                .map(|axis| gradient[axis] * node[axis])
                .sum::<Scalar>()
        })
        .collect()
}

fn forces(element: &Element, values: &[Scalar]) -> Vec<Scalar> {
    element
        .nodal_forces(&MODEL, &temperatures(values))
        .unwrap()
        .iter()
        .map(|force| force.value())
        .collect()
}

fn potential(element: &Element, values: &[Scalar]) -> Scalar {
    element
        .potential(&MODEL, &temperatures(values))
        .unwrap()
        .value()
}

fn stiffnesses(element: &Element, values: &[Scalar]) -> Vec<Vec<Scalar>> {
    element
        .nodal_stiffnesses(&MODEL, &temperatures(values))
        .unwrap()
        .iter()
        .map(|row| row.iter().map(|entry| entry.value()).collect())
        .collect()
}

fn assert_close(a: Scalar, b: Scalar, tolerance: Scalar) {
    crate::math::assert::Assert {
        abs_tol: tolerance,
        rel_tol: tolerance,
        ..Default::default()
    }
    .eq_within_tols(a, &b)
    .unwrap()
}

#[test]
fn gradient_of_a_linear_field_is_its_slope() {
    let gradient = [0.3, -0.7, 1.1];
    cases(0.1).iter().for_each(|(element, nodes)| {
        let found = element.temperature_gradient(&temperatures(&linear(nodes, gradient)));
        (0..3).for_each(|axis| assert_close(found[axis].value(), gradient[axis], 1e-12))
    })
}

#[test]
fn forces_of_a_linear_field_do_not_depend_on_the_stabilization() {
    let gradient = [0.3, -0.7, 1.1];
    cases(0.1)
        .iter()
        .zip(cases(0.9).iter())
        .for_each(|((low, nodes), (high, _))| {
            let values = linear(nodes, gradient);
            forces(low, &values)
                .iter()
                .zip(forces(high, &values))
                .for_each(|(&a, b)| assert_close(a, b, 1e-12))
        })
}

#[test]
fn potential_of_a_linear_field_does_not_depend_on_the_stabilization() {
    let gradient = [0.3, -0.7, 1.1];
    cases(0.1)
        .iter()
        .zip(cases(0.9).iter())
        .for_each(|((low, nodes), (high, _))| {
            let values = linear(nodes, gradient);
            assert_close(potential(low, &values), potential(high, &values), 1e-12)
        })
}

#[test]
fn forces_sum_to_zero() {
    cases(0.3).iter().for_each(|(element, nodes)| {
        let total: Scalar = forces(element, &pseudorandom(nodes.len())).iter().sum();
        assert_close(total, 0.0, 1e-12)
    })
}

#[test]
fn forces_are_the_gradient_of_the_potential() {
    cases(0.3).iter().for_each(|(element, nodes)| {
        let values = pseudorandom(nodes.len());
        let found = forces(element, &values);
        (0..nodes.len()).for_each(|node| {
            let shifted = |shift: Scalar| {
                let mut values = values.clone();
                values[node] += shift;
                potential(element, &values)
            };
            let finite_difference = (shifted(EPSILON) - shifted(-EPSILON)) / (2.0 * EPSILON);
            assert_close(found[node], finite_difference, 1e-7)
        })
    })
}

#[test]
fn stiffnesses_are_the_gradient_of_the_forces() {
    cases(0.3).iter().for_each(|(element, nodes)| {
        let values = pseudorandom(nodes.len());
        let found = stiffnesses(element, &values);
        (0..nodes.len()).for_each(|node| {
            let shifted = |shift: Scalar| {
                let mut values = values.clone();
                values[node] += shift;
                forces(element, &values)
            };
            let (above, below) = (shifted(EPSILON), shifted(-EPSILON));
            (0..nodes.len()).for_each(|other| {
                assert_close(
                    found[other][node],
                    (above[other] - below[other]) / (2.0 * EPSILON),
                    1e-7,
                )
            })
        })
    })
}

#[test]
fn stiffnesses_are_symmetric_and_annihilate_constants() {
    cases(0.3).iter().for_each(|(element, nodes)| {
        let found = stiffnesses(element, &pseudorandom(nodes.len()));
        (0..nodes.len()).for_each(|row| {
            assert_close(found[row].iter().sum(), 0.0, 1e-12);
            (0..nodes.len())
                .for_each(|column| assert_close(found[row][column], found[column][row], 1e-12))
        })
    })
}

#[test]
fn stiffnesses_are_positive_semidefinite() {
    cases(0.3).iter().for_each(|(element, nodes)| {
        let found = stiffnesses(element, &pseudorandom(nodes.len()));
        (0..4).for_each(|trial| {
            let mode = (0..nodes.len())
                .map(|index| ((index + 3 * trial) as Scalar * 7.1).cos())
                .collect::<Vec<_>>();
            let energy: Scalar = found
                .iter()
                .zip(&mode)
                .map(|(row, &a)| a * row.iter().zip(&mode).map(|(&k, &b)| k * b).sum::<Scalar>())
                .sum();
            assert!(energy > 0.0, "{energy}")
        })
    })
}

#[test]
fn stabilization_resists_the_hourglass_mode_in_proportion() {
    let mode = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    let energy = |stabilization| {
        let (element, nodes) = box_(stabilization);
        stiffnesses(&element, &pseudorandom(nodes.len()))
            .iter()
            .zip(mode)
            .map(|(row, a)| a * row.iter().zip(mode).map(|(&k, b)| k * b).sum::<Scalar>())
            .sum::<Scalar>()
    };
    assert!(energy(0.1) > 1e-3);
    assert_close(energy(0.2) / energy(0.1), 2.0, 1e-9)
}

#[test]
fn box_with_no_stabilization_has_no_hourglass_energy() {
    let (element, nodes) = box_(0.0);
    let found = stiffnesses(&element, &pseudorandom(nodes.len()));
    let mode = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
    found
        .iter()
        .for_each(|row| assert_close(row.iter().zip(mode).map(|(&k, b)| k * b).sum(), 0.0, 1e-12));
    assert!(element.volume().value() > 0.0)
}
