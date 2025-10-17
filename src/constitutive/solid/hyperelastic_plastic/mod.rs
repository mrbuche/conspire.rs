mod hencky;

pub use hencky::Hencky;

use crate::{
    constitutive::{ConstitutiveError, solid::elastic_plastic::ElasticPlastic},
    mechanics::{DeformationGradient, DeformationGradientPlastic, Scalar},
};

/// Required methods for hyperelastic-plastic constitutive models.
pub trait HyperelasticPlastic
where
    Self: ElasticPlastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F}_\mathrm{e})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<Scalar, ConstitutiveError>;
}
