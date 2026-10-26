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
    math::{ContractWith, Quantity, Solution, optimize::OptimizationError},
    units::UnitMul,
};
use std::{fmt::Debug, ops::Mul};

pub(super) trait BacktrackingLineSearch<J, X>
where
    Self: Debug,
{
    type Search: ObjectiveMerit;
    fn backtracking_line_search<D, W, U, V>(
        &self,
        mut function: impl FnMut(&X) -> Result<Quantity<V>, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        argument: &X,
        jacobian0: &J,
        decrement: &D,
        step_size: Quantity<W>,
    ) -> Result<Quantity<W>, OptimizationError>
    where
        J: ContractWith<D, Output = Quantity<U>>,
        W: UnitMul<U, Output = V>,
        X: Solution,
        for<'a> &'a D: Mul<Quantity<W>, Output = X>,
    {
        let initial = jacobian0.contract_with(decrement);
        self.line_search()
            .search(
                |n| function(&(decrement * -n + argument)),
                |n| {
                    if n == Quantity::new(0.0) {
                        Ok(initial)
                    } else {
                        Ok(jacobian(&(decrement * -n + argument))?.contract_with(decrement))
                    }
                },
                step_size,
            )
            .map_err(|error| OptimizationError::upstream(error, self))
    }
    fn line_search(&self) -> &Self::Search;
}
