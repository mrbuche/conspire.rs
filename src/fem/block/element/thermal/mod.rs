pub mod conduction;

use crate::{fem::StandardGradientOperators, math::Scalars};
use std::fmt::{self, Debug, Formatter};

pub struct ThermalElement<const G: usize, const N: usize> {
    integration_weights: Scalars<G>,
    standard_gradient_operators: StandardGradientOperators<3, N, G>,
}

impl<const G: usize, const N: usize> Debug for ThermalElement<G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N) {
            (1, 4) => "LinearTetrahedron",
            (8, 8) => "LinearHexahedron",
            (4, 10) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ Thermal, G: {G}, N: {N} }}",)
    }
}

pub trait ThermalFiniteElement<C, const G: usize, const N: usize>
where
    Self: Debug,
{
    fn integration_weights(&self) -> &Scalars<G>;
    fn standard_gradient_operators(&self) -> &StandardGradientOperators<3, N, G>;
}

impl<C, const G: usize, const N: usize> ThermalFiniteElement<C, G, N> for ThermalElement<G, N> {
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
    fn standard_gradient_operators(&self) -> &StandardGradientOperators<3, N, G> {
        &self.standard_gradient_operators
    }
}
