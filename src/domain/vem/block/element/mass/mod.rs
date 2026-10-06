#[cfg(test)]
mod test;

use crate::{
    math::{Quantity, Scalar, Tensor, TensorRank1, TensorVector},
    units::{Density, Mass},
    vem::{
        NodalReferenceCoordinates,
        block::element::{Element, VirtualElement},
    },
};

pub type ElementNodalLumpedMasses = TensorVector<Quantity<Mass>>;

pub trait LumpedMassVirtualElement
where
    Self: VirtualElement,
{
    fn nodal_lumped_masses(
        &self,
        density: Quantity<Density>,
        nodal_reference_coordinates: &NodalReferenceCoordinates,
    ) -> ElementNodalLumpedMasses;
}

impl LumpedMassVirtualElement for Element {
    fn nodal_lumped_masses(
        &self,
        density: Quantity<Density>,
        nodal_reference_coordinates: &NodalReferenceCoordinates,
    ) -> ElementNodalLumpedMasses {
        let nodes = nodal_reference_coordinates
            .iter()
            .map(array)
            .collect::<Vec<_>>();
        let center = mean(&nodes);
        let tetrahedra = self
            .faces_nodes()
            .iter()
            .flat_map(|face_nodes| {
                let face_center = mean(
                    &face_nodes
                        .iter()
                        .map(|&node| nodes[node])
                        .collect::<Vec<_>>(),
                );
                (0..face_nodes.len())
                    .map(|spot| {
                        [
                            face_center,
                            nodes[face_nodes[(spot + 1) % face_nodes.len()]],
                            nodes[face_nodes[spot]],
                            center,
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let volume = self.integration_weights()[0];
        let gradients = self.gradient_vectors()[0]
            .iter()
            .map(array)
            .collect::<Vec<_>>();
        let mass = density * volume;
        diagonal_scaling(&nodes, center, &tetrahedra, &gradients, volume.value())
            .into_iter()
            .map(|fraction| mass * fraction)
            .collect()
    }
}

fn array<I, U>(vector: &TensorRank1<3, I, U>) -> [Scalar; 3] {
    [vector[0].value(), vector[1].value(), vector[2].value()]
}

fn mean(points: &[[Scalar; 3]]) -> [Scalar; 3] {
    let number = points.len() as Scalar;
    [0, 1, 2].map(|axis| points.iter().map(|point| point[axis]).sum::<Scalar>() / number)
}

fn diagonal_scaling(
    nodes: &[[Scalar; 3]],
    center: [Scalar; 3],
    tetrahedra: &[[[Scalar; 3]; 4]],
    gradients: &[[Scalar; 3]],
    volume: Scalar,
) -> Vec<Scalar> {
    let dot = |a: &[Scalar; 3], b: &[Scalar; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let minus = |a: &[Scalar; 3], b: &[Scalar; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut first = [0.0; 3];
    let mut second = [[0.0; 3]; 3];
    let mut signed_volume = 0.0;
    tetrahedra.iter().for_each(|tetrahedron| {
        let [d_0, d_1, d_2] = [0, 1, 2].map(|index| minus(&tetrahedron[index], &center));
        let cross = [
            d_1[1] * d_2[2] - d_1[2] * d_2[1],
            d_1[2] * d_2[0] - d_1[0] * d_2[2],
            d_1[0] * d_2[1] - d_1[1] * d_2[0],
        ];
        let tetrahedron_volume = dot(&d_0, &cross) / 6.0;
        signed_volume += tetrahedron_volume;
        let sum = [
            d_0[0] + d_1[0] + d_2[0],
            d_0[1] + d_1[1] + d_2[1],
            d_0[2] + d_1[2] + d_2[2],
        ];
        (0..3).for_each(|i| {
            first[i] += tetrahedron_volume / 4.0 * sum[i];
            (0..3).for_each(|j| {
                second[i][j] += tetrahedron_volume / 20.0
                    * (d_0[i] * d_0[j] + d_1[i] * d_1[j] + d_2[i] * d_2[j] + sum[i] * sum[j])
            })
        })
    });
    if signed_volume * volume < 0.0 {
        first.iter_mut().for_each(|entry| *entry = -*entry);
        second
            .iter_mut()
            .flatten()
            .for_each(|entry| *entry = -*entry);
    }
    let number_of_nodes = nodes.len() as Scalar;
    let offsets = nodes
        .iter()
        .map(|node| minus(node, &center))
        .collect::<Vec<_>>();
    let diagonal = gradients
        .iter()
        .zip(&offsets)
        .map(|(gradient, offset)| {
            let second_moment = (0..3)
                .map(|i| gradient[i] * (0..3).map(|j| second[i][j] * gradient[j]).sum::<Scalar>())
                .sum::<Scalar>();
            let projection = |other: &[Scalar; 3]| 1.0 / number_of_nodes + dot(gradient, other);
            let stabilization = 1.0 - 2.0 * projection(offset)
                + offsets
                    .iter()
                    .map(|other| projection(other).powi(2))
                    .sum::<Scalar>();
            volume / (number_of_nodes * number_of_nodes)
                + 2.0 / number_of_nodes * dot(gradient, &first)
                + second_moment
                + volume * stabilization
        })
        .collect::<Vec<_>>();
    let trace = diagonal.iter().sum::<Scalar>();
    diagonal.into_iter().map(|entry| entry / trace).collect()
}
