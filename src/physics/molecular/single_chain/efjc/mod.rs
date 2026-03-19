#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, special::erf},
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
            Thermodynamics,
            ufjc::{
                // nondimensional_compliance as nondimensional_compliance_asymptotic,
                nondimensional_extension as nondimensional_extension_asymptotic,
                nondimensional_gibbs_free_energy_per_link as nondimensional_gibbs_free_energy_per_link_asymptotic,
            },
        },
    },
};
use std::f64::consts::PI;

/// The extensible freely-jointed chain model.[^1]<sup>,</sup>[^2]
/// [^1]: N. Balabaev and T. Khazanovich, [Russian Journal of Physical Chemistry B  **3**, 242 (2009)](https://doi.org/10.1134/S1990793109020109).
/// [^2]: M.R. Buche, M.N. Silberstein, and S.J. Grutzik, [Physical Review E **106**, 024502 (2022)](https://doi.org/10.1103/PhysRevE.106.024502).
#[derive(Clone, Debug)]
pub struct ExtensibleFreelyJointedChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The link stiffness $`k_b`$.
    pub link_stiffness: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl ExtensibleFreelyJointedChain {
    fn nondimensional_link_stiffness(&self) -> Scalar {
        self.link_stiffness * self.link_length().powi(2) / BOLTZMANN_CONSTANT / self.temperature()
    }
}

impl SingleChain for ExtensibleFreelyJointedChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Thermodynamics for ExtensibleFreelyJointedChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for ExtensibleFreelyJointedChain {
    fn nondimensional_helmholtz_free_energy(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_force(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_stiffness(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl Isotensional for ExtensibleFreelyJointedChain {
    /// ```math
    /// \varrho(\eta) = \varrho_a(\eta) - \ln\left[1 + g(\eta)\right]
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let eta_exp = eta.exp();
        Ok(nondimensional_gibbs_free_energy_per_link_asymptotic(
            eta,
            kappa,
            -0.5 * eta.powi(2) / kappa,
            1.0,
        )? - (0.5
            + ((eta_over_kappa + 1.0) * eta_exp * erf(&((eta + kappa) / (2.0 * kappa).sqrt()))
                - (eta_over_kappa - 1.0) / eta_exp * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
                / (4.0 * eta.sinh() * (1.0 + eta / eta.tanh() / kappa)))
            .ln())
    }
    /// ```math
    /// \gamma(\eta) = \gamma_a(\eta) + \frac{g'(\eta)}{1 + g(\eta)}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let denominator = 4.0 * eta.sinh() * (1.0 + eta / eta.tanh() / kappa);
        let fraction = ((eta_over_kappa + 1.0)
            * eta.exp()
            * erf(&((eta + kappa) / (2.0 * kappa).sqrt()))
            - (eta_over_kappa - 1.0) / eta.exp() * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
            / denominator;
        Ok(
            nondimensional_extension_asymptotic(eta, kappa, eta_over_kappa, 1.0)?
                + (eta.exp()
                    * ((2.0 / PI / kappa).sqrt()
                        * (eta_over_kappa + 1.0)
                        * (-(eta + kappa).powi(2) / 2.0 / kappa).exp()
                        + (1.0 + (1.0 + eta) / kappa))
                    - 1.0 / eta.exp()
                        * ((2.0 / PI / kappa).sqrt()
                            * (eta_over_kappa - 1.0)
                            * (-(eta - kappa).powi(2) / 2.0 / kappa).exp()
                            + (1.0 + (1.0 - eta) / kappa)
                                * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
                    - fraction
                        * (4.0
                            * (eta.cosh() * (1.0 + (1.0 + eta / eta.tanh()) / kappa)
                                - eta_over_kappa / eta.sinh())))
                    / denominator
                    / (1.0 + fraction),
        )
    }
    /// ```math
    /// \zeta(\eta) = \zeta_a(\eta) + \frac{g''(\eta)}{1 + g(\eta)} - \left[\frac{g'(\eta)}{1 + g(\eta)}\right]^2
    /// ```
    fn nondimensional_compliance(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl Legendre for ExtensibleFreelyJointedChain {
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}
