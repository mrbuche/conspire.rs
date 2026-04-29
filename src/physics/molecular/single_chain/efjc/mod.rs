#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, random_uniform, random_x2_normal, special::erf},
    mechanics::CurrentCoordinate,
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Configuration, Ensemble, Extensible, Isometric, Isotensional, IsotensionalExtensible,
            Legendre, MonteCarlo, SingleChain, SingleChainError, Thermodynamics,
            ThermodynamicsExtensible,
            ufjc::{
                // nondimensional_compliance as nondimensional_compliance_asymptotic,
                nondimensional_extension as nondimensional_extension_asymptotic,
                nondimensional_gibbs_free_energy_per_link as nondimensional_gibbs_free_energy_per_link_asymptotic,
                nondimensional_link_energy_average as nondimensional_link_energy_average_asymptotic,
                nondimensional_link_energy_variance as nondimensional_link_energy_variance_asymptotic,
            },
        },
    },
};
use std::f64::consts::{PI, TAU};

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

impl Extensible for ExtensibleFreelyJointedChain {}

impl Thermodynamics for ExtensibleFreelyJointedChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl ThermodynamicsExtensible for ExtensibleFreelyJointedChain {}

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
    /// \varrho(\eta) = \ln\left[\frac{\eta}{\sinh(\eta)}\right] - \ln\left[1 + \frac{\eta}{c\kappa}\,\coth(\eta)\right] - \nu(\eta) - \ln\left[1 + g(\eta)\right]
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let neg_2_eta_exp = (-2.0 * eta).exp();
        Ok(nondimensional_gibbs_free_energy_per_link_asymptotic(
            eta,
            kappa,
            -0.5 * eta.powi(2) / kappa,
            1.0,
        )? - (0.5
            + ((eta_over_kappa + 1.0) * erf(&((eta + kappa) / (2.0 * kappa).sqrt()))
                - (eta_over_kappa - 1.0)
                    * neg_2_eta_exp
                    * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
                / (2.0 * (1.0 - neg_2_eta_exp) * (1.0 + eta / eta.tanh() / kappa)))
            .ln())
    }
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta) + \frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{c + (\eta/\kappa)\coth(\eta)}\right] + \Delta\lambda(\eta) + \frac{g'(\eta)}{1 + g(\eta)}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let neg_2_eta_exp = (-2.0 * eta).exp();
        let denominator = 2.0 * (1.0 - neg_2_eta_exp) * (1.0 + eta / eta.tanh() / kappa);
        let fraction = ((eta_over_kappa + 1.0) * erf(&((eta + kappa) / (2.0 * kappa).sqrt()))
            - (eta_over_kappa - 1.0)
                * neg_2_eta_exp
                * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
            / denominator;
        Ok(
            nondimensional_extension_asymptotic(eta, kappa, eta_over_kappa, 1.0)?
                + (((2.0 / PI / kappa).sqrt()
                    * (eta_over_kappa + 1.0)
                    * (-(eta + kappa).powi(2) / 2.0 / kappa).exp()
                    + (1.0 + (1.0 + eta) / kappa))
                    - 1.0
                        * neg_2_eta_exp
                        * ((2.0 / PI / kappa).sqrt()
                            * (eta_over_kappa - 1.0)
                            * (-(eta - kappa).powi(2) / 2.0 / kappa).exp()
                            + (1.0 + (1.0 - eta) / kappa)
                                * erf(&((eta - kappa) / (2.0 * kappa).sqrt())))
                    - fraction
                        * (2.0
                            * ((1.0 + neg_2_eta_exp) * (1.0 + (1.0 + eta / eta.tanh()) / kappa)
                                - 4.0 * eta_over_kappa / (1.0 / neg_2_eta_exp - 1.0))))
                    / denominator
                    / (1.0 + fraction),
        )
    }
    /// ```math
    /// \zeta(\eta) = \mathcal{L}(\eta) + \frac{\partial}{\partial\eta}\left\{\frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{c + (\eta/\kappa)\coth(\eta)}\right]\right\} + \zeta(\eta) + \frac{g''(\eta)}{1 + g(\eta)} - \left[\frac{g'(\eta)}{1 + g(\eta)}\right]^2
    /// ```
    fn nondimensional_compliance(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl IsotensionalExtensible for ExtensibleFreelyJointedChain {
    /// ```math
    /// \langle\upsilon\rangle = \frac{1}{2} + \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)} + \frac{\eta^2}{2\kappa} + \frac{g'(\upsilon)}{1 + g(\upsilon)}
    /// ```
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let upsilon = 0.5 * eta.powi(2) / kappa;

        let eta_over_kappa = eta / kappa;
        let neg_2_eta_exp = (-2.0 * eta).exp();
        let eta_coth = 1.0 / eta.tanh();

        let sqrt_2_kappa = (2.0 * kappa).sqrt();
        let x_plus = (eta + kappa) / sqrt_2_kappa;
        let x_minus = (eta - kappa) / sqrt_2_kappa;

        let erf_plus = erf(&x_plus);
        let erf_minus = erf(&x_minus);

        let a =
            (eta_over_kappa + 1.0) * erf_plus - (eta_over_kappa - 1.0) * neg_2_eta_exp * erf_minus;

        let d = 2.0 * (1.0 - neg_2_eta_exp) * (1.0 + eta_over_kappa * eta_coth);

        let f = 0.5 + a / d;

        let dx_plus_dkappa = (kappa - eta) / (2.0 * kappa).powf(1.5);
        let dx_minus_dkappa = -(kappa + eta) / (2.0 * kappa).powf(1.5);

        let derf_plus_dkappa = (2.0 / PI.sqrt()) * (-(x_plus.powi(2))).exp() * dx_plus_dkappa;
        let derf_minus_dkappa = (2.0 / PI.sqrt()) * (-(x_minus.powi(2))).exp() * dx_minus_dkappa;

        let da_dkappa = -eta / kappa.powi(2) * erf_plus
            + (eta_over_kappa + 1.0) * derf_plus_dkappa
            + eta / kappa.powi(2) * neg_2_eta_exp * erf_minus
            - (eta_over_kappa - 1.0) * neg_2_eta_exp * derf_minus_dkappa;

        let dd_dkappa = -2.0 * (1.0 - neg_2_eta_exp) * eta * eta_coth / kappa.powi(2);

        let df_dkappa = (da_dkappa * d - a * dd_dkappa) / d.powi(2);

        Ok(
            nondimensional_link_energy_average_asymptotic(eta, kappa, upsilon, 1.0)?
                - kappa * df_dkappa / f,
        )
    }
    /// ```math
    /// \sigma_\upsilon^2 = \frac{1}{2} + \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)}\left[2 - \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)}\right] + ???
    /// ```
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!()
    }
    fn nondimensional_link_energy_probability(
        &self,
        nondimensional_energy: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("Need to calculate the TSTs and add to uFJC.")
    }
    fn nondimensional_link_length_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("Need to calculate the TSTs and add to uFJC.")
    }
    fn nondimensional_link_length_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("Need to calculate the TSTs and add to uFJC.")
    }
    fn nondimensional_link_length_probability(
        &self,
        nondimensional_length: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("Need to calculate the TSTs and add to uFJC.")
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

impl MonteCarlo for ExtensibleFreelyJointedChain {
    fn random_nondimensional_link_vectors(&self, nondimensional_force: Scalar) -> Configuration {
        let sigma = 1.0 / self.nondimensional_link_stiffness().sqrt();
        (0..self.number_of_links())
            .map(|_| {
                let cos_theta = if nondimensional_force == 0.0 {
                    2.0 * random_uniform() - 1.0
                } else {
                    todo!("Force biases the link stretch too.")
                };
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi = TAU * random_uniform();
                let (sin_phi, cos_phi) = phi.sin_cos();
                let lambda = random_x2_normal(1.0, sigma);
                CurrentCoordinate::from([
                    lambda * sin_theta * cos_phi,
                    lambda * sin_theta * sin_phi,
                    lambda * cos_theta,
                ])
            })
            .collect()
    }
}
