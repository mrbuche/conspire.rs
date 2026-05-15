#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, Tensor, random_uniform, random_x2_normal},
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
    fn nondimensional_longitudinal_extension(
        &self,
        nondimensional_force: Scalar,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> Scalar {
        nondimensional_extension_reweighted_biased_stretch(
            self,
            nondimensional_force,
            nondimensional_force,
            number_of_samples,
            number_of_threads,
        )
    }
    fn random_nondimensional_link_vectors(&self, nondimensional_force: Scalar) -> Configuration {
        if nondimensional_force != 0.0 {
            unimplemented!()
        }
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
                &b * random_x2_normal(1.0, std)
            })
            .collect()
    }
}

fn random_nondimensional_link_vectors_biased_stretch(
    model: &ExtensibleFreelyRotatingChain,
    nondimensional_stretch_bias: Scalar,
) -> Configuration {
    let kappa = model.nondimensional_link_stiffness();
    let std = 1.0 / kappa.sqrt();
    let mean = 1.0 + nondimensional_stretch_bias / kappa;

    let cos_theta = 2.0 * random_uniform() - 1.0;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = TAU * random_uniform();
    let (sin_phi, cos_phi) = phi.sin_cos();

    const AY: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 1.0, 0.0]);
    const AZ: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 0.0, 1.0]);

    let mut a = AY;
    let mut b =
        CurrentCoordinate::const_from([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]);

    let (sin_theta, cos_theta) = model.link_angle.sin_cos();

    (0..model.number_of_links())
        .map(|link| {
            if link > 0 {
                a = if b[1].abs() < 0.9 { AY } else { AZ };
                let u = a.cross(&b).normalized();
                let v = b.cross(&u);
                let phi = TAU * random_uniform();
                let (sin_phi, cos_phi) = phi.sin_cos();
                b = &b * cos_theta + (&u * cos_phi + &v * sin_phi) * sin_theta;
            }
            &b * random_x2_normal(mean, std)
        })
        .collect()
}

use std::thread::scope;

fn nondimensional_extension_reweighted_biased_stretch(
    model: &ExtensibleFreelyRotatingChain,
    nondimensional_force: Scalar,
    nondimensional_stretch_bias: Scalar,
    number_of_samples: usize,
    number_of_threads: usize,
) -> Scalar {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;

    scope(|s| {
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_extension_reweighted_biased_stretch_inner(
                        model,
                        nondimensional_force,
                        nondimensional_stretch_bias,
                        base + usize::from(t < remainder),
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .reduce(|mut acc, (x_max, z_scaled, ext_scaled)| {
                let x_max_new = acc.0.max(x_max);
                let scale_acc = (acc.0 - x_max_new).exp();
                let scale_new = (x_max - x_max_new).exp();

                acc.1 = acc.1 * scale_acc + z_scaled * scale_new;
                acc.2 = acc.2 * scale_acc + ext_scaled * scale_new;
                acc.0 = x_max_new;
                acc
            })
            .map(|(_x_max, z_scaled, ext_scaled)| {
                ext_scaled / z_scaled / model.number_of_links() as Scalar
            })
            .unwrap()
    })
}

fn nondimensional_extension_reweighted_biased_stretch_inner(
    model: &ExtensibleFreelyRotatingChain,
    nondimensional_force: Scalar,
    nondimensional_stretch_bias: Scalar,
    number_of_samples: usize,
) -> (Scalar, Scalar, Scalar) {
    let mut x_max = Scalar::NEG_INFINITY;
    let mut z_scaled = 0.0;
    let mut ext_scaled = 0.0;

    for _ in 0..number_of_samples {
        let links =
            random_nondimensional_link_vectors_biased_stretch(model, nondimensional_stretch_bias);

        let extension_sum: Scalar = links.iter().map(|link| link[2]).sum();
        let stretch_sum: Scalar = links.iter().map(|link| link.norm()).sum();

        let x = nondimensional_force * extension_sum - nondimensional_stretch_bias * stretch_sum;

        if x > x_max {
            let scale = if x_max.is_finite() {
                (x_max - x).exp()
            } else {
                0.0
            };
            z_scaled *= scale;
            ext_scaled *= scale;
            x_max = x;
        }

        let w = (x - x_max).exp();
        z_scaled += w;
        ext_scaled += extension_sum * w;
    }

    (x_max, z_scaled, ext_scaled)
}
