use super::{LocalSupport, Subdomain};
use crate::math::Vector;

pub(crate) struct CornerSelection {
    corners: Vec<usize>,
}

impl CornerSelection {
    pub(crate) fn new(corners: Vec<usize>) -> Self {
        Self { corners }
    }
    pub(crate) fn corners(&self) -> &[usize] {
        &self.corners
    }
}

pub(crate) struct DualPrimalSplit {
    primal: Vec<usize>,
    dual: Vec<usize>,
}

impl DualPrimalSplit {
    fn from_corners(_corners: &CornerSelection, _num_dofs: usize) -> Self {
        todo!("partition subdomain DOFs into corner-primal and remaining-dual sets")
    }
    pub(crate) fn primal(&self) -> &[usize] {
        &self.primal
    }
    pub(crate) fn dual(&self) -> &[usize] {
        &self.dual
    }
}

pub(crate) fn solve<B>(_subdomains: &[Subdomain<B>], _corners: &CornerSelection) -> Vector
where
    B: LocalSupport,
{
    todo!(
        "assemble the corner coarse problem, condense to non-singular local K_s, projected PCG on the dual DOFs"
    )
}
