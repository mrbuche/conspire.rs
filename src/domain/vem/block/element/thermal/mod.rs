pub mod conduction;

use crate::{
    fem::block::element::thermal::ElementNodalTemperatures as TetrahedronNodalTemperatures,
    math::{Quantity, Scalar, Tensor},
    mechanics::TemperatureGradient,
    units::Temperature,
    vem::block::element::{Element, VirtualElement},
};

pub use crate::domain::thermal::NodalTemperatures as ElementNodalTemperatures;

pub trait ThermalVirtualElement
where
    Self: VirtualElement,
{
    fn temperature_gradient(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> TemperatureGradient;
    fn tetrahedra_temperatures(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Vec<TetrahedronNodalTemperatures<4>>;
}

fn mean(temperatures: Vec<Quantity<Temperature>>) -> Quantity<Temperature> {
    let number = temperatures.len() as Scalar;
    temperatures.into_iter().sum::<Quantity<Temperature>>() / number
}

impl ThermalVirtualElement for Element {
    fn temperature_gradient(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> TemperatureGradient {
        nodal_temperatures
            .iter()
            .zip(&self.gradient_vectors()[0])
            .map(|(nodal_temperature, gradient_vector)| gradient_vector * nodal_temperature)
            .sum()
    }
    fn tetrahedra_temperatures(
        &self,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Vec<TetrahedronNodalTemperatures<4>> {
        let element_center = mean(nodal_temperatures.iter().copied().collect());
        let faces_centers = self
            .faces_nodes()
            .iter()
            .map(|face_nodes| {
                mean(
                    face_nodes
                        .iter()
                        .map(|&node| nodal_temperatures[node])
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        self.tetrahedra_nodes()
            .iter()
            .map(|&[face, node_b, node_a]| {
                [
                    faces_centers[face],
                    nodal_temperatures[node_b],
                    nodal_temperatures[node_a],
                    element_center,
                ]
                .into()
            })
            .collect()
    }
}
