use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    domain::time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
    fem::{
        Blocks, ElementModelError, Elements, Model,
        block::{
            Block,
            element::{
                FiniteElementError,
                thermal::{
                    capacity::LumpedHeatCapacityFiniteElement,
                    conduction::ThermalConductionFiniteElement,
                },
            },
            thermal::{NodalTemperatures, ThermalElements, capacity::HeatCapacities},
        },
    },
    math::{Quantity, Tensor},
    units::Time,
};

/// Elements that report how fast their diffusion can be.
pub trait ThermalTimeScaleElements
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given temperatures, the reciprocal of
    /// the largest eigenvalue of the lumped heat capacity inverted against the conduction
    /// tangent, which bounds a stable explicit time step.
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> ThermalTimeScaleElements for Model<B, D>
where
    B: ThermalTimeScaleElements,
{
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks.fastest_diffusive_time_scale(nodal_temperatures)
    }
}

impl<B1, B2> ThermalTimeScaleElements for Blocks<B1, B2>
where
    B1: ThermalTimeScaleElements,
    B2: ThermalTimeScaleElements,
{
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_diffusive_time_scale(nodal_temperatures)?
            .min(self.1.fastest_diffusive_time_scale(nodal_temperatures)?))
    }
}

impl<C, F, R, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalTimeScaleElements for Block<C, F, G, M, N, P, R>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P> + LumpedHeatCapacityFiniteElement<G, M, N, P>,
    R: HeatCapacities<G>,
{
    fn fastest_diffusive_time_scale(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, element_connectivity))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                    )?;
                    let capacities: Vec<f64> = element
                        .nodal_lumped_heat_capacities(&self.density().at(element_index))
                        .iter()
                        .map(|capacity| capacity.value())
                        .collect();
                    let eigenvalue = largest_eigenvalue(
                        N,
                        |row, column| stiffnesses[row][column].value(),
                        &capacities,
                    );
                    Ok::<_, FiniteElementError>(
                        fastest.min(diffusive_time_scale_from_eigenvalue(eigenvalue)),
                    )
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
