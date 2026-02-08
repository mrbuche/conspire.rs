//! Plastic fluid constitutive models.

use crate::{
    constitutive::ConstitutiveError,
    math::{Scalar, TensorTuple, TensorTupleVec},
    mechanics::DeformationGradientPlastic,
};
use std::fmt::Debug;

/// Plastic state variables.
pub type StateVariables = TensorTuple<DeformationGradientPlastic, Scalar>;

/// Plastic state variables history.
pub type StateVariablesHistory = TensorTupleVec<DeformationGradientPlastic, Scalar>;

/// Required methods for plastic fluid constitutive models.
pub trait Plastic
where
    Self: Clone + Debug,
{
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Scalar;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Scalar;
    /// Calculates and returns the yield stress.
    ///
    /// ```math
    /// Y = Y_0 + H\,\varepsilon_\mathrm{p}
    /// ```
    fn yield_stress(&self, equivalent_plastic_strain: Scalar) -> Result<Scalar, ConstitutiveError> {
        //
        // Can eventually make a subdirectory with an enum (like LineaSearch) with different hardening models.
        //
        Ok(self.initial_yield_stress() + self.hardening_slope() * equivalent_plastic_strain)
    }
}
