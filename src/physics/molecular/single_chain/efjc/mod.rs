#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar,
        special::{langevin, langevin_derivative},
    },
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
            Thermodynamics,
            ufjc::{
                nondimensional_extension as nondimensional_extension_asymptotic,
                nondimensional_gibbs_free_energy_per_link as nondimensional_gibbs_free_energy_per_link_asymptotic,
            },
        },
    },
};

/// The extensible freely-jointed chain model.
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
    /// \varrho(\eta) = ???
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        nondimensional_gibbs_free_energy_per_link_asymptotic(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            -0.5 * nondimensional_force.powi(2) / self.nondimensional_link_stiffness(),
            1.0,
        )
    }
    /// ```math
    /// \gamma(\eta) = ???
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        nondimensional_extension_asymptotic(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            nondimensional_force / self.nondimensional_link_stiffness(),
            1.0,
        )
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
            let delta_lambda = eta / kappa;
            let c_0 = langevin_derivative(eta);
            let g = 1.0 - gamma_0 * eta_coth;
            let h = 1.0 + delta_lambda * eta_coth;
            let dcth = 1.0 - 1.0 / (eta_tanh * eta_tanh);
            let dg = -(c_0 * eta_coth + gamma_0 * dcth);
            let dh = eta_coth / kappa + delta_lambda * dcth;
            Ok(c_0 + (1.0 + g / h) / kappa + delta_lambda * (dg * h - g * dh) / (h * h))
        }
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
