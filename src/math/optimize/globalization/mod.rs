mod line_search;
mod trust_region;

pub use line_search::{
    Armijo, Error, Goldstein, LineSearchError, NoSearch, ObjectiveMerit, Residual, ResidualMerit,
    Wolfe,
};
pub use trust_region::TrustRegion;
pub(crate) use trust_region::{
    TRUST_REGION_ETA, TRUST_REGION_MAX_REJECTIONS, TRUST_REGION_SHRINK, more_sorensen_dense,
    more_sorensen_sparse, update_radius,
};

use crate::{
    math::{
        Erase, Jacobian, Scalar, Solution, Tensor,
        optimize::{OptimizationError, StepSize},
    },
    units::UnitDiv,
};
use std::{fmt::Debug, ops::Mul};

pub(super) trait BacktrackingLineSearch<J, X>
where
    Self: Debug,
{
    type Search: ObjectiveMerit;
    fn backtracking_line_search<D, E>(
        &self,
        function: impl FnMut(&X, Scalar) -> Result<Scalar, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Scalar,
    ) -> Result<Scalar, OptimizationError>
    where
        J: Erase<Erased = E> + Jacobian,
        D: Erase<Erased = E> + Tensor,
        E: Tensor,
        X: Solution,
        <X as Tensor>::Unit: UnitDiv<<D as Tensor>::Unit>,
        for<'a> &'a D: Mul<StepSize<D, X>, Output = X>,
    {
        self.line_search()
            .search(
                function, jacobian, argument, jacobian0, decrement, step_size,
            )
            .map_err(|error| OptimizationError::upstream(error, self))
    }
    fn line_search(&self) -> &Self::Search;
}
