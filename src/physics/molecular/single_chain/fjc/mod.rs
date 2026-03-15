#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar,
        special::{inverse_langevin, langevin, langevin_derivative},
    },
    physics::molecular::single_chain::{
        Ensemble, Inextensible, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
        Thermodynamics,
    },
};
use std::f64::consts::PI;

/// The freely-jointed chain model.
#[derive(Clone, Debug)]
pub struct FreelyJointedChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for FreelyJointedChain {
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for FreelyJointedChain {
    fn maximum_nondimensional_extension(&self) -> Scalar {
        1.0
    }
}

impl Thermodynamics for FreelyJointedChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for FreelyJointedChain {
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(0.0)
        } else {
            let [s0, _, _] = treloar_sums(self.number_of_links(), nondimensional_extension);
            Ok(nondimensional_extension.abs().ln() - s0.ln())
        }
    }
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(0.0)
        } else {
            let [s0, s1, _] = treloar_sums(self.number_of_links(), nondimensional_extension);
            let n = self.number_of_links() as Scalar;
            Ok((1.0 / nondimensional_extension + (0.5 * n - 1.0) * s1 / s0) / n)
        }
    }
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(Scalar::NAN)
        } else {
            let [s0, s1, s2] =
                treloar_sums(self.number_of_links(), nondimensional_extension);

            if !s0.is_finite() || s0 == 0.0 {
                return Ok(Scalar::NAN);
            }

            let n = self.number_of_links() as Scalar;
            let p = n - 2.0;
            let b = (0.5 * n - 1.0) / n;

            let ds0dx = -(p / 2.0) * s1;
            let ds1dx = -((p - 1.0) / 2.0) * s2;
            let d_ratio_dx = (ds1dx * s0 - s1 * ds0dx) / (s0 * s0);

            Ok(-1.0 / (n * nondimensional_extension * nondimensional_extension)
                + b * d_ratio_dx)
        }
    }
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension <= 0.0 || nondimensional_extension >= 1.0 {
            Ok(0.0)
        } else {
            let number_of_links = self.number_of_links();
            let [s0, _, _] = treloar_sums(number_of_links, nondimensional_extension);
            let n = number_of_links as Scalar;
            let factorial_n_minus_2 = (1..=(number_of_links - 2))
                .map(|i| i as Scalar)
                .product::<Scalar>();
            Ok((n.powf(n) / (8.0 * PI * nondimensional_extension * factorial_n_minus_2)) * s0)
        }
    }
}

impl Isotensional for FreelyJointedChain {
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(self.number_of_links() as Scalar
            * (nondimensional_force / nondimensional_force.sinh()).ln())
    }
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(langevin(nondimensional_force))
    }
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(langevin_derivative(nondimensional_force))
    }
}

impl Legendre for FreelyJointedChain {
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        Ok(inverse_langevin(nondimensional_extension))
    }
}

fn treloar_sums(num_links: u8, x: Scalar) -> [Scalar; 3] {
    if num_links <= 2 {
        return [Scalar::NAN; 3];
    }
    let n = num_links as Scalar;
    let p = (num_links - 2) as i32;
    let m = 0.5 * (1.0 - x);
    let k = ((n * m).ceil() as usize)
        .saturating_sub(1)
        .min(num_links as usize);
    let k_float = n * m;
    if (k_float - k_float.round()).abs() == 0.0 {
        return [Scalar::NAN; 3];
    }
    let mut binom = 1.0;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for s in 0..=k {
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        let t = m - (s as Scalar) / n;
        let t0 = if p >= 0 {
            t.powi(p)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p)
        };
        let t1 = if p - 1 >= 0 {
            t.powi(p - 1)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p - 1)
        };
        let t2 = if p - 2 >= 0 {
            t.powi(p - 2)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p - 2)
        };
        s0 += sign * binom * t0;
        s1 += sign * binom * t1;
        s2 += sign * binom * t2;
        let sf = s as Scalar;
        binom *= (n - sf) / (sf + 1.0);
    }
    [s0, s1, s2]
}
