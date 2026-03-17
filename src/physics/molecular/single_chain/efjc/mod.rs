#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, special::sinhc},
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
            Thermodynamics,
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
    pub fn nondimensional_link_stiffness(&self, temperature: Scalar) -> Scalar {
        self.link_stiffness * self.link_length.powi(2) / BOLTZMANN_CONSTANT / temperature
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
    /// \beta\varphi(\eta) = ???
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        //
        // uFJC impl will use this and an enum for potentials
        // then put exact here for EFJC
        // and separate helper functions for the common terms between both
        //
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness(temperature);
        Ok(self.number_of_links() as Scalar
            * -((sinhc(eta) * (1.0 + eta / kappa / eta.tanh())).ln() + 0.5 * eta.powi(2) / kappa))
    }
    /// ```math
    /// \gamma(\eta) = ???
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        if nondimensional_force == 0.0 {
            Ok(0.0)
        } else {
            let eta = nondimensional_force;
            let kappa = self.nondimensional_link_stiffness(temperature);
            let eta_coth = 1.0 / eta.tanh();
            let gamma_0 = eta_coth - 1.0 / eta;
            let delta_lambda = eta / kappa;
            Ok(gamma_0
                + delta_lambda
                    * (1.0 + (1.0 - gamma_0 * eta_coth) / (1.0 + delta_lambda * eta_coth)))
        }
    }
    /// ```math
    /// c(\eta) = ???
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        let kappa = self.nondimensional_link_stiffness(temperature);
        if nondimensional_force == 0.0 {
            Ok(1.0 / 3.0 + (5.0 / 3.0 * kappa + 1.0) / kappa / (kappa + 1.0))
        } else {
            let eta = nondimensional_force;
            // Helpful shorthands
            let t = eta.tanh();
            let cth = 1.0 / t;
            let inv_eta = 1.0 / eta;
            let inv_eta2 = inv_eta * inv_eta;

            // gamma0 = coth(eta) - 1/eta
            let gamma0 = cth - inv_eta;

            // delta = eta/kappa
            let delta = eta / kappa;

            // q = coth(eta)
            let q = cth;

            // g = 1 - gamma0*coth(eta) = 1 - gamma0*q
            let g = 1.0 - gamma0 * q;

            // h = 1 + delta*coth(eta) = 1 + delta*q
            let h = 1.0 + delta * q;

            // Derivatives:
            // d/deta coth(eta) = -csch^2(eta) = -(1/sinh^2(eta)) = -(1/tanh^2(eta) - 1)
            let dcth = 1.0 - 1.0 / (t * t); // = -(1/t^2 - 1)

            // gamma0' = coth'(eta) + 1/eta^2
            let dgamma0 = dcth + inv_eta2;

            // delta' = 1/kappa
            let ddelta = 1.0 / kappa;

            // g' = -(gamma0'*q + gamma0*q')
            let dg = -(dgamma0 * q + gamma0 * dcth);

            // h' = delta'*q + delta*q'
            let dh = ddelta * q + delta * dcth;

            // gamma = gamma0 + delta * (1 + g/h)
            // compliance c = gamma' = gamma0' + delta'*(1+g/h) + delta*(g/h)'
            // (g/h)' = (g'*h - g*h')/h^2
            let one_plus_g_over_h = 1.0 + g / h;
            let d_g_over_h = (dg * h - g * dh) / (h * h);

            let compliance = dgamma0 + ddelta * one_plus_g_over_h + delta * d_g_over_h;

            Ok(compliance)
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
