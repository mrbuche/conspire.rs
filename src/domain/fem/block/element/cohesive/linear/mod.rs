mod wedge;

use crate::fem::block::element::cohesive::CohesiveElement;

pub use wedge::Wedge;

pub type LinearCohesiveElement<const G: usize, const N: usize> = CohesiveElement<G, N, 1>;
