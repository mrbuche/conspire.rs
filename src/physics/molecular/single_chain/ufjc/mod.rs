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
            Ensemble, Extensible, Isometric, Isotensional, IsotensionalExtensible, Legendre,
            SingleChain, SingleChainError, Thermodynamics, ThermodynamicsExtensible,
        },
    },
};
use std::f64::consts::TAU;

/// The freely-jointed chain model with an arbitrary link potential.[^1]
/// [^1]: M.R. Buche, M.N. Silberstein, and S.J. Grutzik, [Physical Review E **106**, 024502 (2022)](https://doi.org/10.1103/PhysRevE.106.024502).
#[derive(Clone, Debug)]
pub struct ArbitraryPotentialFreelyJointedChain<T>
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

impl<T> ArbitraryPotentialFreelyJointedChain<T>
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

impl<T> SingleChain for ArbitraryPotentialFreelyJointedChain<T>
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

impl<T> Extensible for ArbitraryPotentialFreelyJointedChain<T> where T: Potential {}

impl<T> Thermodynamics for ArbitraryPotentialFreelyJointedChain<T>
where
    T: Potential,
{
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl<T> ThermodynamicsExtensible for ArbitraryPotentialFreelyJointedChain<T> where T: Potential {}

impl<T> Isometric for ArbitraryPotentialFreelyJointedChain<T>
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

impl<T> Isotensional for ArbitraryPotentialFreelyJointedChain<T>
where
    T: Potential,
{
    /// ```math
    /// \varrho(\eta) = \ln\left[\frac{\eta}{\sinh(\eta)}\right] - \ln\left[1 + \frac{\eta}{c\kappa}\,\coth(\eta)\right] - \nu(\eta)
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        nondimensional_gibbs_free_energy_per_link(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            self.link_potential
                .nondimensional_legendre(nondimensional_force, self.temperature()),
            self.correction(),
        )
    }
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta) + \frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{c + (\eta/\kappa)\coth(\eta)}\right] + \Delta\lambda(\eta)
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        nondimensional_extension(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            self.link_potential
                .nondimensional_extension(nondimensional_force, self.temperature()),
            self.correction(),
        )
    }
    /// ```math
    /// \zeta(\eta) = \mathcal{L}(\eta) + \frac{\partial}{\partial\eta}\left\{\frac{\eta}{\kappa}\left[\frac{1 - \mathcal{L}(\eta)\coth(\eta)}{c + (\eta/\kappa)\coth(\eta)}\right]\right\} + \zeta(\eta)
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        nondimensional_compliance(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            self.link_potential
                .nondimensional_compliance(nondimensional_force, self.temperature()),
            self.correction(),
        )
    }
}

impl<T> IsotensionalExtensible for ArbitraryPotentialFreelyJointedChain<T>
where
    T: Potential,
{
    /// ```math
    /// \langle\upsilon\rangle = \frac{1}{2} + \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)} + \upsilon[\lambda(\eta)]
    /// ```
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(0.5
            + helper(
                nondimensional_force,
                self.nondimensional_link_stiffness(),
                self.correction(),
            )
            + self
                .link_potential
                .nondimensional_energy_at_nondimensional_force(
                    nondimensional_force,
                    self.temperature(),
                ))
    }
    /// ```math
    /// \sigma_\upsilon^2(\eta) = \frac{1}{2} + \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)}\left[2 - \frac{\eta/\kappa}{\eta/\kappa + c\tanh(\eta)}\right] + ???
    /// ```
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        //
        // Need to match last term correctly for nonlinear potentials.
        //
        let hlpr = helper(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            self.correction(),
        );
        Ok(0.5
            + hlpr * (2.0 - hlpr)
            + nondimensional_force.powi(2) / self.nondimensional_link_stiffness())
    }
    /// ```math
    /// p(\upsilon\,|\,\eta) = \left|\frac{\partial\upsilon}{\partial\lambda}\right|^{-1} \Big[p(\lambda_+\,|\,\eta) + p(\lambda_-\,|\,\eta)\Big]
    /// ```
    fn nondimensional_link_energy_probability(
        &self,
        nondimensional_energy: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.link_potential
            .nondimensional_forces_at_nondimensional_energy(
                nondimensional_energy,
                self.temperature(),
            )
            .into_iter()
            .zip(
                self.link_potential
                    .nondimensional_lengths_at_nondimensional_energy(
                        nondimensional_energy,
                        self.temperature(),
                    ),
            )
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
    /// ```math
    /// \langle\lambda\rangle = 1 + \frac{1}{\kappa}\big[1 + \eta\coth(\eta)\big] + ???
    /// ```
    fn nondimensional_link_length_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        //
        // Need to match last term correctly for nonlinear potentials.
        //
        todo!()
    }
    /// ```math
    /// \sigma_\lambda^2 = \frac{1}{\kappa} + ???
    /// ```
    fn nondimensional_link_length_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        //
        // Need to match last term correctly for nonlinear potentials.
        //
        Ok(1.0 / self.nondimensional_link_stiffness())
    }
    /// ```math
    /// p(\lambda\,|\,\eta) = \left(\frac{2\pi}{\kappa}\right)^{-1/2}\frac{\mathrm{sinhc}(\lambda\eta)}{\mathrm{sinhc}(\eta)}\,\frac{e^{-\upsilon(\lambda)}\,e^{-\eta^2/2\kappa}}{1 + (\eta/c\kappa)\coth(\eta)}
    /// ```
    fn nondimensional_link_length_probability(
        &self,
        nondimensional_length: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let kappa = self.nondimensional_link_stiffness();
        let lambda = nondimensional_length;
        let eta = nondimensional_force;
        let upsilon_twice = kappa * (lambda - 1.0).powi(2) / 2.0 + eta.powi(2) / 2.0 / kappa;
        Ok((kappa / TAU).sqrt()
            * lambda
            * ((eta * (lambda - 1.0) - upsilon_twice).exp()
                - (-eta * (lambda + 1.0) - upsilon_twice).exp())
            / (1.0 - (-2.0 * eta).exp())
            / (1.0 + eta / kappa / eta.tanh()))
    }
}

fn helper(
    nondimensional_force: Scalar,
    nondimensional_stiffness: Scalar,
    correction: Scalar,
) -> Scalar {
    let eta_over_kappa = nondimensional_force / nondimensional_stiffness;
    eta_over_kappa / (eta_over_kappa + correction * nondimensional_force.tanh())
}

impl<T> Legendre for ArbitraryPotentialFreelyJointedChain<T>
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

pub fn nondimensional_gibbs_free_energy_per_link(
    eta: Scalar,
    kappa: Scalar,
    nu: Scalar,
    c: Scalar,
) -> Result<Scalar, SingleChainError> {
    Ok(-((sinhc(eta) * (1.0 + eta / c / kappa / eta.tanh())).ln() - nu))
}

pub fn nondimensional_extension(
    eta: Scalar,
    kappa: Scalar,
    delta_lambda: Scalar,
    c: Scalar,
) -> Result<Scalar, SingleChainError> {
    if eta == 0.0 {
        Ok(0.0)
    } else {
        let eta_coth = 1.0 / eta.tanh();
        let gamma_0 = langevin(eta);
        let eta_over_kappa = eta / kappa;
        Ok(gamma_0
            + eta_over_kappa * (1.0 - gamma_0 * eta_coth) / (c + eta_over_kappa * eta_coth)
            + delta_lambda)
    }
}

pub fn nondimensional_compliance(
    eta: Scalar,
    kappa: Scalar,
    zeta: Scalar,
    c: Scalar,
) -> Result<Scalar, SingleChainError> {
    if eta == 0.0 {
        Ok(1.0 / 3.0 + 2.0 / 3.0 / c / kappa + zeta)
    } else {
        let eta_tanh = eta.tanh();
        let eta_coth = 1.0 / eta_tanh;
        let gamma_0 = langevin(eta);
        let eta_over_kappa = eta / kappa;
        let c_0 = langevin_derivative(eta);
        let g = 1.0 - gamma_0 * eta_coth;
        let h = c + eta_over_kappa * eta_coth;
        let dcth = 1.0 - 1.0 / (eta_tanh * eta_tanh);
        let dg = -(c_0 * eta_coth + gamma_0 * dcth);
        let dh = eta_coth / kappa + eta_over_kappa * dcth;
        Ok(c_0 + (g / h) / kappa + eta_over_kappa * (dg * h - g * dh) / (h * h) + zeta)
    }
}
