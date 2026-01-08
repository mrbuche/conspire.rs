// pub mod linear;

use crate::math::ScalarList;

pub struct CohesiveElement<const G: usize, const N: usize, const O: usize> {
    integration_weights: ScalarList<G>,
}

// no gradient vectors, and shape functions (and their gradients) are known for arbitrary elements
// do others really integrate in the current configuration?

// no deformation gradients (and not a solid), so will not need to store the reference normal either

// do not do piecewise-linear wedge-12 one, just have the linear wedge-6s placed for that
