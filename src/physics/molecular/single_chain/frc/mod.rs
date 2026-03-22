#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, Tensor, TensorArray},
    mechanics::CurrentCoordinate,
    physics::molecular::single_chain::{
        Configuration, Ensemble, Inextensible, MonteCarlo, SingleChain,
    },
    random_uniform,
};
use std::f64::consts::TAU;

/// The freely-rotating chain model.
#[derive(Clone, Debug)]
pub struct FreelyRotatingChain {
    /// The link angle $`\theta_b`$.
    pub link_angle: Scalar,
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for FreelyRotatingChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for FreelyRotatingChain {
    fn maximum_nondimensional_extension(&self) -> Scalar {
        1.0
    }
}

impl MonteCarlo for FreelyRotatingChain {
    fn random_configuration(&self) -> Configuration {
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
                position += &b;
                position.clone()
            })
            .collect()
    }
}
