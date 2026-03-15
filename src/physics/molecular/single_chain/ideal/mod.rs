#[cfg(test)]
mod test;

use crate::{
    math::Scalar,
    physics::molecular::single_chain::{
        Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError, Thermodynamics,
    },
};
use std::f64::consts::PI;

/// The ideal chain model.
#[derive(Clone, Debug)]
pub struct IdealChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for IdealChain {
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Thermodynamics for IdealChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for IdealChain {
    /// ```math
    /// \beta\psi(\gamma) = \frac{3}{2}\,N_b\gamma^2
    /// ```
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(1.5 * self.number_of_links() as Scalar * nondimensional_extension.powi(2))
    }
    /// ```math
    /// \eta(\gamma) = 3\gamma
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(3.0 * nondimensional_extension)
    }
    /// ```math
    /// k(\gamma) = 3
    /// ```
    fn nondimensional_stiffness(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(3.0)
    }
    /// ```math
    /// \mathcal{P}(\gamma) = \left(\frac{3}{2\pi N_b}\right)^{3/2}\exp\left(-\frac{3}{2}\,N_b\gamma^2\right)
    /// ```
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let number_of_links = self.number_of_links() as Scalar;
        Ok((1.5 / PI / number_of_links).powf(1.5)
            * (-1.5 * number_of_links * nondimensional_extension.powi(2)).exp())
    }
}

impl Isotensional for IdealChain {
    /// ```math
    /// \beta\varphi(\eta) = \frac{1}{6}\,N_b\eta^2
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(self.number_of_links() as Scalar / -6.0 * nondimensional_force.powi(2))
    }
    /// ```math
    /// \gamma(\eta) = \frac{\eta}{3}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(nondimensional_force / 3.0)
    }
    /// ```math
    /// c(\eta) = \frac{1}{3}
    /// ```
    fn nondimensional_compliance(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(1.0 / 3.0)
    }
}

impl Legendre for IdealChain {
    /// ```math
    /// \eta(\gamma) = 3\gamma
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(3.0 * nondimensional_extension)
    }
    /// ```math
    /// \gamma(\eta) = \frac{\eta}{3}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(nondimensional_force / 3.0)
    }
}
