#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar,
        special::{langevin, langevin_derivative, sinhc},
    },
    physics::molecular::{
        potential::Potential,
        single_chain::{
            Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
            Thermodynamics,
        },
    },
};

/// The extensible freely-jointed chain model.
#[derive(Clone, Debug)]
pub struct Foo<T>
where
    T: Potential,
{
    /// The link potential $`u`$.
    pub link_potential: T,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl<T> Foo<T>
where
    T: Potential,
{
    fn correction(&self) -> Scalar {
        1.0 / (1.0
            - 0.5
                * self
                    .link_potential
                    .nondimensional_anharmonicity(1.0, self.temperature())
                / self
                    .link_potential
                    .nondimensional_stiffness(1.0, self.temperature()))
    }
    fn nondimensional_link_stiffness(&self) -> Scalar {
        self.link_potential
            .nondimensional_stiffness(0.0, self.temperature())
    }
}

impl<T> SingleChain for Foo<T>
where
    T: Potential,
{
    fn link_length(&self) -> Scalar {
        self.link_potential.rest_length()
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl<T> Thermodynamics for Foo<T>
where
    T: Potential,
{
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl<T> Isometric for Foo<T>
where
    T: Potential,
{
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

impl<T> Isotensional for Foo<T>
where
    T: Potential,
{
    /// ```math
    /// \varrho(\eta) = \ln\left[\frac{\eta}{\sinh(\eta)}\right] - \ln\left[1 + \frac{\eta}{c\kappa}\,\coth(\eta)\right] - \beta v(\eta)
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness();
        let beta_v = self
            .link_potential
            .nondimensional_legendre(eta, self.temperature());
        let c = self.correction();
        Ok(-((sinhc(eta) * (1.0 + eta / c / kappa / eta.tanh())).ln() - beta_v))
    }
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta) + \frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{c + (\eta/\kappa)\coth(\eta)}\right] + \Delta\lambda(\eta)
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        if nondimensional_force == 0.0 {
            Ok(0.0)
        } else {
            let eta = nondimensional_force;
            let eta_coth = 1.0 / eta.tanh();
            let gamma_0 = langevin(eta);
            let kappa = self.nondimensional_link_stiffness();
            let eta_over_kappa = eta / kappa;
            let delta_lambda = self
                .link_potential
                .nondimensional_extension(eta, self.temperature());
            let c = self.correction();
            Ok(gamma_0
                + eta_over_kappa * (1.0 - gamma_0 * eta_coth) / (c + eta_over_kappa * eta_coth)
                + delta_lambda)
        }
    }
    /// ```math
    /// c(\eta) = ???
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let kappa = self.nondimensional_link_stiffness();
        if nondimensional_force == 0.0 {
            Ok(1.0 / 3.0 + (5.0 / 3.0 * kappa + 1.0) / kappa / (kappa + 1.0))
        } else {
            let eta = nondimensional_force;
            let eta_tanh = eta.tanh();
            let eta_coth = 1.0 / eta_tanh;
            let gamma_0 = langevin(eta);
            let eta_over_kappa = eta / kappa;
            let c_0 = langevin_derivative(eta);
            let g = 1.0 - gamma_0 * eta_coth;
            let h = 1.0 + eta_over_kappa * eta_coth;
            let dcth = 1.0 - 1.0 / (eta_tanh * eta_tanh);
            let dg = -(c_0 * eta_coth + gamma_0 * dcth);
            let dh = eta_coth / kappa + eta_over_kappa * dcth;
            let p = self
                .link_potential
                .nondimensional_compliance(eta, self.temperature());
            let c = self.correction();
            Ok(c_0 + (g / h) / kappa + eta_over_kappa * (dg * h - g * dh) / (h * h) + p)
        }
    }
}

impl<T> Legendre for Foo<T>
where
    T: Potential,
{
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}
