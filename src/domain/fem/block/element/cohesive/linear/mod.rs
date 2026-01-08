mod wedge;

use crate::fem::block::element::{
    FiniteElement,
    cohesive::{CohesiveElement, M},
};

pub use wedge::Wedge;

pub type LinearCohesiveElement<const G: usize, const N: usize, const P: usize> = CohesiveElement<G, N, 1, P>;
