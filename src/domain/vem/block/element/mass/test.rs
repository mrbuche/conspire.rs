use crate::{
    math::{Quantity, Scalar, Tensor},
    mechanics::ReferenceCoordinate,
    units::{Density, Mass},
    vem::{
        NodalReferenceCoordinates,
        block::element::{
            Element, ElementNodalReferenceCoordinates, VirtualElement,
            mass::LumpedMassVirtualElement,
        },
    },
};

type Case = (Element, NodalReferenceCoordinates);

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

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
        Element::from((coordinates, &faces_indices[..], &nodes_indices[..], faces)),
        nodal_coordinates,
    )
}

fn fractions((element, coordinates): &Case) -> Vec<Scalar> {
    let total = DENSITY * element.integration_weights()[0];
    element
        .nodal_lumped_masses(DENSITY, coordinates)
        .iter()
        .map(|mass| (*mass / total).value())
        .collect()
}

fn tetrahedron() -> Case {
    element(
        &[
            [0.1, 0.2, 0.0],
            [1.3, 0.1, 0.2],
            [0.2, 0.9, 0.1],
            [0.3, 0.4, 1.2],
        ],
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
    )
}

fn cube() -> Case {
    element(
        &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ],
        &[
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![3, 7, 6, 2],
            vec![0, 4, 7, 3],
            vec![1, 2, 6, 5],
        ],
    )
}

fn bipyramid(height: Scalar, below: Scalar) -> Case {
    let s = 3.0_f64.sqrt() / 2.0;
    bipyramid_at(&[
        [1.0, 0.0, 0.0],
        [-0.5, s, 0.0],
        [-0.5, -s, 0.0],
        [0.0, 0.0, height],
        [0.0, 0.0, -below],
    ])
}

fn bipyramid_at(nodes: &[[Scalar; 3]; 5]) -> Case {
    element(
        nodes,
        &[
            vec![0, 1, 3],
            vec![1, 2, 3],
            vec![2, 0, 3],
            vec![1, 0, 4],
            vec![2, 1, 4],
            vec![0, 2, 4],
        ],
    )
}

fn assert_close(a: Scalar, b: Scalar, tolerance: Scalar) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn shares_a_skewed_tetrahedron_equally() {
    fractions(&tetrahedron())
        .iter()
        .for_each(|&fraction| assert_close(fraction, 0.25, 1e-12));
}

#[test]
fn shares_a_box_equally() {
    fractions(&cube())
        .iter()
        .for_each(|&fraction| assert_close(fraction, 0.125, 1e-12));
}

#[test]
fn totals_the_mass_of_the_element() {
    [tetrahedron(), cube(), bipyramid(0.7, 0.4)]
        .iter()
        .for_each(|(element, coordinates)| {
            let total = element
                .nodal_lumped_masses(DENSITY, coordinates)
                .iter()
                .copied()
                .sum::<Quantity<Mass>>();
            assert_close(
                (total / (DENSITY * element.integration_weights()[0])).value(),
                1.0,
                1e-12,
            )
        })
}

#[test]
fn respects_the_symmetry_of_a_bipyramid() {
    let fractions = fractions(&bipyramid(0.7, 0.7));
    assert_close(fractions[0], fractions[1], 1e-12);
    assert_close(fractions[1], fractions[2], 1e-12);
    assert_close(fractions[3], fractions[4], 1e-12);
    fractions
        .iter()
        .for_each(|&fraction| assert!(fraction > 0.0));
}

#[test]
fn stays_positive_and_bounded_on_a_flat_agglomerate() {
    let fractions: Vec<Vec<Scalar>> = [1e-1, 1e-3, 1e-5]
        .iter()
        .map(|&epsilon| fractions(&bipyramid(epsilon, 1.0)))
        .collect();
    fractions.iter().for_each(|fractions| {
        fractions
            .iter()
            .for_each(|&fraction| assert!(fraction > 1e-3, "{fractions:?}"))
    });
    fractions[1]
        .iter()
        .zip(&fractions[2])
        .for_each(|(&a, &b)| assert_close(a, b, 1e-3));
}

#[test]
fn matches_a_quadrature_of_the_projection_on_a_lopsided_bipyramid() {
    let s = 3.0_f64.sqrt() / 2.0;
    let nodes = [
        [1.0, 0.0, 0.0],
        [-0.5, s, 0.0],
        [-0.5, -s, 0.0],
        [0.1, 0.2, 0.9],
        [-0.2, 0.1, -0.4],
    ];
    let case = bipyramid_at(&nodes);
    let element = &case.0;
    let number_of_nodes = nodes.len() as Scalar;
    let center =
        [0, 1, 2].map(|axis| nodes.iter().map(|node| node[axis]).sum::<Scalar>() / number_of_nodes);
    let gradients = (0..nodes.len())
        .map(|node| [0, 1, 2].map(|axis| element.gradient_vectors()[0][node][axis].value()))
        .collect::<Vec<_>>();
    let (a, b) = (0.585_410_196_624_968_5, 0.138_196_601_125_010_5);
    let rule = [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]];
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]];
    let volume = element.integration_weights()[0].value();
    let projection = |gradient: &[Scalar; 3], point: &[Scalar; 3]| {
        1.0 / number_of_nodes
            + (0..3)
                .map(|axis| gradient[axis] * (point[axis] - center[axis]))
                .sum::<Scalar>()
    };
    let diagonal = gradients
        .iter()
        .enumerate()
        .map(|(node, gradient)| {
            let consistent = tetrahedra
                .iter()
                .map(|tetrahedron| {
                    let [p0, p1, p2, p3] = tetrahedron.map(|index| nodes[index]);
                    let edges = [0, 1, 2].map(|axis| {
                        [
                            p1[axis] - p0[axis],
                            p2[axis] - p0[axis],
                            p3[axis] - p0[axis],
                        ]
                    });
                    let determinant = edges[0][0]
                        * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
                        - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
                        + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0]);
                    let tetrahedron_volume = determinant.abs() / 6.0;
                    rule.iter()
                        .map(|weights| {
                            let point = [0, 1, 2].map(|axis| {
                                weights[0] * p0[axis]
                                    + weights[1] * p1[axis]
                                    + weights[2] * p2[axis]
                                    + weights[3] * p3[axis]
                            });
                            tetrahedron_volume / 4.0 * projection(gradient, &point).powi(2)
                        })
                        .sum::<Scalar>()
                })
                .sum::<Scalar>();
            let stabilization = 1.0 - 2.0 * projection(gradient, &nodes[node])
                + nodes
                    .iter()
                    .map(|other| projection(gradient, other).powi(2))
                    .sum::<Scalar>();
            consistent + volume * stabilization
        })
        .collect::<Vec<_>>();
    let trace = diagonal.iter().sum::<Scalar>();
    fractions(&case)
        .iter()
        .zip(&diagonal)
        .for_each(|(&fraction, &entry)| assert_close(fraction, entry / trace, 1e-12));
}
