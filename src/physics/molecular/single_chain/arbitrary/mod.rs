use crate::{
    math::{Scalar, random::random_uniform},
    mechanics::CurrentCoordinate,
    physics::molecular::{
        potential::{Harmonic, Potential},
        single_chain::{
            Configuration, Ensemble, Isometric, Isotensional, Legendre, MonteCarlo, SingleChain,
            SingleChainError, Thermodynamics,
        },
    },
};
use std::f64::consts::TAU;

/// Options for arbitrary discrete potentials.
#[derive(Clone, Debug)]
pub enum ArbitraryDiscretePotential<U>
where
    U: Potential,
{
    Free,
    Rigid(Scalar),
    Strong(U),
    Weak(U),
}

/// The arbitrary discrete single-chain model.
#[derive(Clone, Debug)]
pub struct ArbitraryDiscrete {
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The link potential $`u_b`$.
    pub link_potential: ArbitraryDiscretePotential<Harmonic>,
    /// The angular potential $`u_\theta`$.
    pub angular_potential: ArbitraryDiscretePotential<Harmonic>,
    /// The torsional potential $`u_\phi`$.
    pub torsional_potential: ArbitraryDiscretePotential<Harmonic>,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for ArbitraryDiscrete {
    fn link_length(&self) -> Scalar {
        match &self.link_potential {
            ArbitraryDiscretePotential::Free => panic!(),
            ArbitraryDiscretePotential::Rigid(link_length) => *link_length,
            ArbitraryDiscretePotential::Strong(link_potential) => link_potential.rest_length(),
            ArbitraryDiscretePotential::Weak(link_potential) => link_potential.rest_length(),
        }
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Thermodynamics for ArbitraryDiscrete {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for ArbitraryDiscrete {
    fn nondimensional_helmholtz_free_energy(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_force(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_stiffness(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl Isotensional for ArbitraryDiscrete {
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_extension(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
    fn nondimensional_compliance(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl Legendre for ArbitraryDiscrete {}

impl MonteCarlo for ArbitraryDiscrete {
    fn random_nondimensional_link_vectors(&self, nondimensional_force: Scalar) -> Configuration {
        //
        // Need to add cases and get them right.
        //
        let eta = nondimensional_force;
        let eta_exp = eta.exp();
        let eta_nexp = 1.0 / eta_exp;
        (0..self.number_of_links())
            .map(|_| {
                let cos_theta = if eta == 0.0 {
                    2.0 * random_uniform() - 1.0
                } else {
                    (eta_nexp + random_uniform() * (eta_exp - eta_nexp)).ln() / eta
                };
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi = TAU * random_uniform();
                let (sin_phi, cos_phi) = phi.sin_cos();
                CurrentCoordinate::from([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
            })
            .collect()
    }
}
