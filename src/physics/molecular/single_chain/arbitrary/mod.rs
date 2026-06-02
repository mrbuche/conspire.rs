use crate::{
    math::{Scalar, Tensor, random_uniform, random_x2_normal},
    mechanics::CurrentCoordinate,
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::{
            potential::{Cosine, Harmonic, Potential},
            single_chain::{
                Configuration, Ensemble, Isometric, Isotensional, Legendre, MonteCarlo,
                SingleChain, SingleChainError, Thermodynamics,
            },
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
    pub angular_potential: ArbitraryDiscretePotential<Cosine>,
    /// The torsional potential $`u_\phi`$.
    pub torsional_potential: ArbitraryDiscretePotential<Cosine>,
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

impl ArbitraryDiscrete {
    fn frame_propagated_link_vectors(
        &self,
        mut next_cos_theta: impl FnMut() -> Scalar,
        mut next_phi: impl FnMut() -> Scalar,
        mut next_length: impl FnMut() -> Scalar,
    ) -> Configuration {
        const AY: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 1.0, 0.0]);
        const AZ: CurrentCoordinate = CurrentCoordinate::const_from([0.0, 0.0, 1.0]);
        // First link: isotropic random direction.
        let cos_theta = 2.0 * random_uniform() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = TAU * random_uniform();
        let (sin_phi, cos_phi) = phi.sin_cos();
        let mut b =
            CurrentCoordinate::const_from([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]);
        let mut b_previous: Option<CurrentCoordinate> = None;
        (0..self.number_of_links())
            .map(|link| {
                if link > 0 {
                    let axis = b.clone();
                    // Build an orthonormal frame `(e1, e2)` in the plane
                    // perpendicular to `axis`. When a preceding bond exists, its
                    // in-plane projection fixes `e1` so that `phi` is the proper
                    // dihedral angle; otherwise (the second link, or collinear
                    // bonds) the dihedral is undefined, so pick an arbitrary
                    // perpendicular and draw `phi` uniformly.
                    let (e1, phi) = match &b_previous {
                        Some(b_prev) => {
                            let projection = b_prev - &axis * (b_prev * &axis);
                            if projection.norm() > 1e-6 {
                                (projection.normalized(), next_phi())
                            } else {
                                let a = if axis[1].abs() < 0.9 { AY } else { AZ };
                                (a.cross(&axis).normalized(), TAU * random_uniform())
                            }
                        }
                        None => {
                            let a = if axis[1].abs() < 0.9 { AY } else { AZ };
                            (a.cross(&axis).normalized(), TAU * random_uniform())
                        }
                    };
                    let e2 = axis.cross(&e1);
                    let cos_theta = next_cos_theta();
                    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    b = &axis * cos_theta + (&e1 * cos_phi + &e2 * sin_phi) * sin_theta;
                    b_previous = Some(axis);
                }
                // `b` stays a unit direction for propagation; scale on emit.
                &b * next_length()
            })
            .collect()
    }
}

fn random_bend_cos_theta<U>(potential: &U, temperature: Scalar) -> Scalar
where
    U: Potential,
{
    loop {
        let cos_theta = 2.0 * random_uniform() - 1.0;
        let theta = cos_theta.acos();
        let weight = (-potential.energy(theta) / BOLTZMANN_CONSTANT / temperature).exp();
        if random_uniform() < weight {
            return cos_theta;
        }
    }
}

fn random_torsion_phi<U>(potential: &U, temperature: Scalar) -> Scalar
where
    U: Potential,
{
    // The dihedral angle uses the flat measure d(phi), so draw uniformly over
    // [0, 2pi) and accept with the Boltzmann weight exp(-upsilon(phi)).
    loop {
        let phi = TAU * random_uniform();
        let weight = (-potential.energy(phi) / BOLTZMANN_CONSTANT / temperature).exp();
        if random_uniform() < weight {
            return phi;
        }
    }
}

fn random_nondimensional_link_length<U>(
    potential: &ArbitraryDiscretePotential<U>,
    temperature: Scalar,
) -> Scalar
where
    U: Potential,
{
    match potential {
        ArbitraryDiscretePotential::Free => panic!(),
        ArbitraryDiscretePotential::Rigid(_) => 1.0,
        ArbitraryDiscretePotential::Strong(potential)
        | ArbitraryDiscretePotential::Weak(potential) => {
            let sigma = 1.0 / potential.nondimensional_stiffness(1.0, temperature).sqrt();
            random_x2_normal(1.0, sigma)
        }
    }
}

impl MonteCarlo for ArbitraryDiscrete {
    fn random_nondimensional_link_vectors(&self, nondimensional_force: Scalar) -> Configuration {
        let temperature = self.temperature();
        // Fast path: with free bend and free torsion the links are independent
        // and isotropic (freely-jointed), so they can be drawn directly and
        // biased by the applied force along the z-axis. Force biasing of the
        // direction is only valid for inextensible (rigid) links; an extensible
        // link under force couples stretch and orientation (see EFJC), which is
        // not yet handled.
        if matches!(self.angular_potential, ArbitraryDiscretePotential::Free)
            && matches!(self.torsional_potential, ArbitraryDiscretePotential::Free)
        {
            let eta = nondimensional_force;
            if eta != 0.0 && !matches!(self.link_potential, ArbitraryDiscretePotential::Rigid(_)) {
                unimplemented!()
            }
            let eta_exp = eta.exp();
            let eta_nexp = 1.0 / eta_exp;
            return (0..self.number_of_links())
                .map(|_| {
                    let cos_theta = if eta == 0.0 {
                        2.0 * random_uniform() - 1.0
                    } else {
                        (eta_nexp + random_uniform() * (eta_exp - eta_nexp)).ln() / eta
                    };
                    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                    let phi = TAU * random_uniform();
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    let lambda =
                        random_nondimensional_link_length(&self.link_potential, temperature);
                    CurrentCoordinate::from([
                        lambda * sin_theta * cos_phi,
                        lambda * sin_theta * sin_phi,
                        lambda * cos_theta,
                    ])
                })
                .collect();
        }
        // Otherwise the bend and/or torsion couple neighbouring links, so build
        // the chain by propagating a frame. Force biasing is not yet handled.
        if nondimensional_force != 0.0 {
            unimplemented!()
        }
        self.frame_propagated_link_vectors(
            || match &self.angular_potential {
                ArbitraryDiscretePotential::Free => 2.0 * random_uniform() - 1.0,
                ArbitraryDiscretePotential::Rigid(link_angle) => link_angle.cos(),
                ArbitraryDiscretePotential::Strong(potential)
                | ArbitraryDiscretePotential::Weak(potential) => {
                    random_bend_cos_theta(potential, temperature)
                }
            },
            || match &self.torsional_potential {
                ArbitraryDiscretePotential::Free => TAU * random_uniform(),
                ArbitraryDiscretePotential::Rigid(dihedral_angle) => *dihedral_angle,
                ArbitraryDiscretePotential::Strong(potential)
                | ArbitraryDiscretePotential::Weak(potential) => {
                    random_torsion_phi(potential, temperature)
                }
            },
            || random_nondimensional_link_length(&self.link_potential, temperature),
        )
    }
}
