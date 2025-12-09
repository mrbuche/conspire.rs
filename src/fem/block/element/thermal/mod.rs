pub mod conduction;

use crate::fem::block::element::Element;
use std::fmt::{self, Debug, Formatter};

pub type ThermalElement<const G: usize, const N: usize> = Element<G, [bool; N]>;

pub trait ThermalFiniteElement<C, const G: usize, const N: usize>
where
    Self: Debug,
{
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
