use crate::{
    math::Scalar,
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{Ensemble, SingleChain},
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

// Need to make all Thermodynamics/etc. methods functions of the temperature too.
