#[cfg(test)]
mod test;

use crate::{
    math::Scalar,
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
            Thermodynamics,
            ufjc::{
                nondimensional_compliance as nondimensional_compliance_asymptotic,
                nondimensional_extension as nondimensional_extension_asymptotic,
                nondimensional_gibbs_free_energy_per_link as nondimensional_gibbs_free_energy_per_link_asymptotic,
            },
        },
    },
};

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
        nondimensional_compliance_asymptotic(
            nondimensional_force,
            self.nondimensional_link_stiffness(),
            1.0 / self.nondimensional_link_stiffness(),
            1.0,
        )
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
