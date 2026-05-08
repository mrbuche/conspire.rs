#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar, random_uniform, random_x2_normal,
        special::{erf, erfc},
    },
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
    /// \varrho(\eta) = \ln\left[\frac{\eta}{\sinh(\eta)}\right] - \ln\left[1 + \frac{\eta}{\kappa}\,\coth(\eta)\right] - \frac{\eta^2}{2\kappa} - \ln\left[1 + g(\eta)\right]
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
            + ((eta_over_kappa + 1.0) * erf((eta + kappa) / (2.0 * kappa).sqrt())
                - (eta_over_kappa - 1.0)
                    * neg_2_eta_exp
                    * erf((eta - kappa) / (2.0 * kappa).sqrt()))
                / (2.0 * (1.0 - neg_2_eta_exp) * (1.0 + eta / eta.tanh() / kappa)))
            .ln())
    }
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta) + \frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{1 + (\eta/\kappa)\coth(\eta)}\right] + \frac{\eta}{\kappa} + \frac{g'(\eta)}{1 + g(\eta)}
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
        let fraction = ((eta_over_kappa + 1.0) * erf((eta + kappa) / (2.0 * kappa).sqrt())
            - (eta_over_kappa - 1.0) * neg_2_eta_exp * erf((eta - kappa) / (2.0 * kappa).sqrt()))
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
                                * erf((eta - kappa) / (2.0 * kappa).sqrt()))
                    - fraction
                        * (2.0
                            * ((1.0 + neg_2_eta_exp) * (1.0 + (1.0 + eta / eta.tanh()) / kappa)
                                - 4.0 * eta_over_kappa / (1.0 / neg_2_eta_exp - 1.0))))
                    / denominator
                    / (1.0 + fraction),
        )
    }
    /// ```math
    /// \zeta(\eta) = \mathcal{L}'(\eta) + \frac{\partial}{\partial\eta}\left\{\frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{1 + (\eta/\kappa)\coth(\eta)}\right]\right\} + \frac{1}{\kappa} + \frac{g''(\eta)}{1 + g(\eta)} - \left[\frac{g'(\eta)}{1 + g(\eta)}\right]^2
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
    /// \langle\upsilon\rangle = \frac{\kappa}{2}\Big(\langle\lambda^2\rangle - 2\langle\lambda\rangle + 1\Big)
    /// ```
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(0.5
            * self.nondimensional_link_stiffness()
            * (nondimensional_link_length_squared_average(
                self.nondimensional_link_stiffness(),
                nondimensional_force,
            )? - 2.0
                * ThermodynamicsExtensible::nondimensional_link_length_average(
                    self,
                    nondimensional_force,
                )?
                + 1.0))
    }
    /// ```math
    /// \sigma_\upsilon^2 = \frac{\kappa^2}{4}\Big(\langle\lambda^4\rangle - 4\langle\lambda^3\rangle + 6\langle\lambda^2\rangle - 4\langle\lambda\rangle + 1\Big) - \langle\upsilon\rangle^2
    /// ```
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(0.25
            * self.nondimensional_link_stiffness().powi(2)
            * (nondimensional_link_length_quad_average(
                self.nondimensional_link_stiffness(),
                nondimensional_force,
            )? - 4.0
                * nondimensional_link_length_cubed_average(
                    self.nondimensional_link_stiffness(),
                    nondimensional_force,
                )?
                + 6.0
                    * nondimensional_link_length_squared_average(
                        self.nondimensional_link_stiffness(),
                        nondimensional_force,
                    )?
                - 4.0
                    * ThermodynamicsExtensible::nondimensional_link_length_average(
                        self,
                        nondimensional_force,
                    )?
                + 1.0)
            - ThermodynamicsExtensible::nondimensional_link_energy_average(
                self,
                nondimensional_force,
            )?
            .powi(2))
    }
    /// ```math
    /// p(\upsilon\,|\,\eta) = \left|\frac{\partial\upsilon}{\partial\lambda}\right|^{-1} \Big[p(\lambda_+\,|\,\eta) + p(\lambda_-\,|\,\eta)\Big]
    /// ```
    fn nondimensional_link_energy_probability(
        &self,
        nondimensional_energy: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let kappa = self.nondimensional_link_stiffness();
        let eta = (2.0 * kappa * nondimensional_energy).sqrt();
        let delta_lambda = (2.0 * nondimensional_energy / kappa).sqrt();
        [eta, -eta]
            .into_iter()
            .zip([1.0 + delta_lambda, 1.0 - delta_lambda])
            .map(|(eta, nondimensional_length)| {
                Ok(
                    IsotensionalExtensible::nondimensional_link_length_probability(
                        self,
                        nondimensional_length,
                        nondimensional_force,
                    )? / eta.abs(),
                )
            })
            .sum()
    }
    fn nondimensional_link_length_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let erfd_p = 1.0 + erf((eta + kappa) / (2.0 * kappa).sqrt());
        let exp_n2_eta_erfc_m = (-2.0 * eta).exp() * erfc((eta - kappa) / (2.0 * kappa).sqrt());
        Ok(
            (4.0 * (-0.5 * (eta.powi(2) / kappa + kappa) - eta).exp() / (TAU * kappa).sqrt()
                * eta_over_kappa
                + (1.0 / kappa + (eta_over_kappa + 1.0).powi(2)) * erfd_p
                - (1.0 / kappa + (eta_over_kappa - 1.0).powi(2)) * exp_n2_eta_erfc_m)
                / ((eta / kappa + 1.0) * erfd_p + (eta / kappa - 1.0) * exp_n2_eta_erfc_m),
        )
    }
    fn nondimensional_link_length_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(nondimensional_link_length_squared_average(
            self.nondimensional_link_stiffness(),
            nondimensional_force,
        )? - ThermodynamicsExtensible::nondimensional_link_length_average(
            self,
            nondimensional_force,
        )?
        .powi(2))
    }
    /// ```math
    /// p(\lambda\,|\,\eta) = \left(\frac{2\pi}{\kappa}\right)^{-1/2}\frac{4\lambda\sinh(\eta\lambda)\,e^{-\upsilon(\lambda)}\,e^{-\eta^2/2\kappa}}{e^\eta(1+\eta/\kappa)(1+\mathrm{erf}_+) - e^{-\eta}(1-\eta/\kappa)(1-\mathrm{erf}_-)}
    /// ```
    fn nondimensional_link_length_probability(
        &self,
        nondimensional_length: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let lambda = nondimensional_length;
        let kappa = self.nondimensional_link_stiffness();
        let eta_over_kappa = eta / kappa;
        let upsilon_twice = 0.5 * kappa * ((lambda - 1.0).powi(2) + eta_over_kappa.powi(2));
        Ok((kappa / TAU).sqrt()
            * 2.0
            * lambda
            * ((eta * (lambda - 1.0) - upsilon_twice).exp()
                - (-eta * (lambda + 1.0) - upsilon_twice).exp())
            / ((1.0 + eta_over_kappa) * (1.0 + erf((eta + kappa) / (2.0 * kappa).sqrt()))
                - (1.0 - eta_over_kappa)
                    * (-2.0 * eta).exp()
                    * erfc((eta - kappa) / (2.0 * kappa).sqrt())))
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

fn nondimensional_link_length_squared_average(
    kappa: Scalar,
    eta: Scalar,
) -> Result<Scalar, SingleChainError> {
    let eta_over_kappa = eta / kappa;
    let erfd_p_pre = (eta / kappa + 1.0) * (1.0 + erf((eta + kappa) / (2.0 * kappa).sqrt()));
    let exp_n2_eta_erfc_m_pre =
        (eta / kappa - 1.0) * (-2.0 * eta).exp() * erfc((eta - kappa) / (2.0 * kappa).sqrt());
    Ok(
        (2.0 * (-0.5 * (eta.powi(2) / kappa + kappa) - eta).exp() / (TAU * kappa).sqrt()
            * ((2.0 / kappa + (eta / kappa + 1.0).powi(2))
                - (2.0 / kappa + (eta / kappa - 1.0).powi(2)))
            + (3.0 / kappa + (eta_over_kappa + 1.0).powi(2)) * erfd_p_pre
            + (3.0 / kappa + (eta_over_kappa - 1.0).powi(2)) * exp_n2_eta_erfc_m_pre)
            / (erfd_p_pre + exp_n2_eta_erfc_m_pre),
    )
}

fn nondimensional_link_length_cubed_average(
    kappa: Scalar,
    eta: Scalar,
) -> Result<Scalar, SingleChainError> {
    let eta_over_kappa = eta / kappa;
    let x_p = (eta + kappa) / (2.0 * kappa).sqrt();
    let x_m = (eta - kappa) / (2.0 * kappa).sqrt();
    let one_plus_erf_p = 1.0 + erf(x_p);
    let erfc_m = erfc(x_m);
    let exp_n2_eta = (-2.0 * eta).exp();
    let denominator =
        (eta_over_kappa + 1.0) * one_plus_erf_p + exp_n2_eta * (eta_over_kappa - 1.0) * erfc_m;
    let p_p = eta.powi(4)
        + 4.0 * eta.powi(3) * kappa
        + 6.0 * eta.powi(2) * kappa * (1.0 + kappa)
        + 4.0 * eta * kappa.powi(2) * (3.0 + kappa)
        + kappa.powi(2) * (3.0 + 6.0 * kappa + kappa.powi(2));
    let p_m = eta.powi(4) - 4.0 * eta.powi(3) * kappa + 6.0 * eta.powi(2) * kappa * (1.0 + kappa)
        - 4.0 * eta * kappa.powi(2) * (3.0 + kappa)
        + kappa.powi(2) * (3.0 + 6.0 * kappa + kappa.powi(2));
    let boundary =
        2.0 * eta * (eta.powi(2) + 5.0 * kappa + 3.0 * kappa.powi(2)) * (2.0 / PI).sqrt()
            / kappa.powf(3.5)
            * (-(eta.powi(2) / (2.0 * kappa) + eta + 0.5 * kappa)).exp();
    let branch_terms = (p_p * one_plus_erf_p - exp_n2_eta * p_m * erfc_m) / kappa.powi(4);
    Ok((boundary + branch_terms) / denominator)
}

fn nondimensional_link_length_quad_average(
    kappa: Scalar,
    eta: Scalar,
) -> Result<Scalar, SingleChainError> {
    let sqrt_kappa = kappa.sqrt();
    let sqrt_2 = 2.0_f64.sqrt();
    let sqrt_pi = PI.sqrt();
    let sqrt_2_pi = (2.0 * PI).sqrt();
    let x_p = (eta + kappa) / (2.0 * kappa).sqrt();
    let x_m = (eta - kappa) / (2.0 * kappa).sqrt();
    let erf_p = erf(x_p);
    let erfc_m = erfc(x_m);
    let exp_p = ((eta + kappa).powi(2) / (2.0 * kappa)).exp();
    let exp_m = ((eta - kappa).powi(2) / (2.0 * kappa)).exp();
    let exp_n2_eta = (-2.0 * eta).exp();
    let denominator =
        (eta / kappa + 1.0) * (1.0 + erf_p) + exp_n2_eta * (eta / kappa - 1.0) * erfc_m;
    let poly_p = eta.powi(5)
        + 5.0 * eta.powi(4) * kappa
        + 10.0 * eta.powi(3) * kappa * (1.0 + kappa)
        + 10.0 * eta.powi(2) * kappa.powi(2) * (3.0 + kappa)
        + 5.0 * eta * kappa.powi(2) * (3.0 + 6.0 * kappa + kappa.powi(2))
        + kappa.powi(3) * (15.0 + 10.0 * kappa + kappa.powi(2));
    let inner = -2.0 * (eta - kappa).powi(4) * sqrt_kappa
        - 18.0 * (eta - kappa).powi(2) * kappa.powf(1.5)
        + 18.0 * kappa.powf(3.5)
        + 2.0 * kappa.powf(4.5)
        + 2.0
            * sqrt_2
            * eta.powi(3)
            * kappa
            * (2.0 * sqrt_2 * sqrt_kappa + 5.0 * exp_p * sqrt_pi + 5.0 * exp_p * kappa * sqrt_pi)
        + eta.powi(5) * exp_p * sqrt_2_pi
        + 15.0 * exp_p * kappa.powi(3) * sqrt_2_pi
        + 10.0 * exp_p * kappa.powi(4) * sqrt_2_pi
        + exp_p * kappa.powi(5) * sqrt_2_pi
        + eta.powi(4) * (2.0 * sqrt_kappa + 5.0 * exp_p * kappa * sqrt_2_pi)
        + 2.0
            * eta.powi(2)
            * kappa.powf(1.5)
            * (9.0
                + 6.0 * kappa
                + 15.0 * exp_p * sqrt_kappa * sqrt_2_pi
                + 5.0 * exp_p * kappa.powf(1.5) * sqrt_2_pi)
        + eta
            * kappa.powi(2)
            * (36.0 * sqrt_kappa
                + 8.0 * kappa.powf(1.5)
                + 15.0 * exp_p * sqrt_2_pi
                + 30.0 * exp_p * kappa * sqrt_2_pi
                + 5.0 * exp_p * kappa.powi(2) * sqrt_2_pi)
        + exp_m * (eta - kappa).powi(5) * sqrt_2_pi * erfc_m
        + 10.0 * exp_m * (eta - kappa).powi(3) * kappa * sqrt_2_pi * erfc_m
        + 15.0 * exp_m * (eta - kappa) * kappa.powi(2) * sqrt_2_pi * erfc_m
        + exp_p * poly_p * sqrt_2_pi * erf_p;
    let numerator = (-(eta.powi(2) / (2.0 * kappa) + eta + 0.5 * kappa)).exp()
        / (TAU.sqrt() * kappa.powi(5))
        * inner;
    Ok(numerator / denominator)
}
