#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, Tensor, TensorArray, random_uniform, random_x2_normal},
    mechanics::CurrentCoordinate,
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::single_chain::{
            Configuration, Ensemble, Extensible, Isometric, Isotensional, Legendre, MonteCarlo,
            SingleChain, SingleChainError, Thermodynamics,
        },
    },
};
use std::f64::consts::TAU;

/// The extensible freely-rotating chain model.
#[derive(Clone, Debug)]
pub struct ExtensibleFreelyRotatingChain {
    /// The link angle $`\theta_b`$.
    pub link_angle: Scalar,
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The link stiffness $`k_b`$.
    pub link_stiffness: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl ExtensibleFreelyRotatingChain {
    fn nondimensional_link_stiffness(&self) -> Scalar {
        self.link_stiffness * self.link_length().powi(2) / BOLTZMANN_CONSTANT / self.temperature()
    }
}

impl SingleChain for ExtensibleFreelyRotatingChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Extensible for ExtensibleFreelyRotatingChain {}

impl Thermodynamics for ExtensibleFreelyRotatingChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for ExtensibleFreelyRotatingChain {
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

impl Isotensional for ExtensibleFreelyRotatingChain {
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

impl Legendre for ExtensibleFreelyRotatingChain {
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl MonteCarlo for ExtensibleFreelyRotatingChain {
    fn random_configuration(&self) -> Configuration {
        let std = 1.0 / self.nondimensional_link_stiffness().sqrt();
        let cos_theta = 2.0 * random_uniform() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = TAU * random_uniform();
        let (sin_phi, cos_phi) = phi.sin_cos();
        const AY: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 1.0, 0.0]);
        const AZ: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 0.0, 1.0]);
        let mut a = AY;
        let mut b =
            CurrentCoordinate::const_from([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]);
        let mut position = CurrentCoordinate::zero();
        let (sin_theta, cos_theta) = self.link_angle.sin_cos();
        (0..self.number_of_links())
            .map(|link| {
                if link > 0 {
                    a = if b[1].abs() < 0.9 { AY } else { AZ };
                    let u = a.cross(&b).normalized();
                    let v = b.cross(&u);
                    let phi = TAU * random_uniform();
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    b = &b * cos_theta + (&u * cos_phi + &v * sin_phi) * sin_theta;
                }
                position += &b * random_x2_normal(1.0, std);
                position.clone()
            })
            .collect()
    }
}
