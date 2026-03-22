#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, TensorArray},
    mechanics::CurrentCoordinate,
    physics::molecular::single_chain::{
        Configuration, Ensemble, Inextensible, Isometric, Isotensional, Legendre, MonteCarlo,
        SingleChain, SingleChainError, Thermodynamics,
    },
    random_uniform,
};
use std::f64::consts::TAU;

/// The square-well freely-jointed chain model.[^1]
/// [^1]: M.R. Buche, M.N. Silberstein, and S.J. Grutzik, [Physical Review E **106**, 024502 (2022)](https://doi.org/10.1103/PhysRevE.106.024502).
#[derive(Clone, Debug)]
pub struct SquareWellFreelyJointedChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The well width $`w_b`$.
    pub well_width: Scalar,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for SquareWellFreelyJointedChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for SquareWellFreelyJointedChain {
    /// ```math
    /// \lim_{\eta\to\infty}\gamma(\eta) = 1 + \frac{w_b}{\ell_b} = \varsigma
    /// ```
    fn maximum_nondimensional_extension(&self) -> Scalar {
        1.0 + self.well_width / self.link_length
    }
}

impl Thermodynamics for SquareWellFreelyJointedChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for SquareWellFreelyJointedChain {
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

impl Isotensional for SquareWellFreelyJointedChain {
    /// ```math
    /// \beta\varphi(\eta) = N_b\ln\left[\frac{\eta^3}{\varsigma\eta\cosh(\varsigma\eta) - \sinh(\varsigma\eta) - \eta\cosh(\eta) + \sinh(\eta)}\right]
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let varsigma = self.maximum_nondimensional_extension();
        let varsigma_eta = varsigma * nondimensional_force;
        Ok(self.number_of_links() as Scalar
            * (nondimensional_force.powi(3)
                / (varsigma_eta * varsigma_eta.cosh()
                    - varsigma_eta.sinh()
                    - nondimensional_force * nondimensional_force.cosh()
                    + nondimensional_force.sinh()))
            .ln())
    }
    /// ```math
    /// \gamma(\eta) = \frac{\varsigma^2\eta\sinh(\varsigma\eta) - \eta\sinh(\eta)}{\varsigma\eta\cosh(\varsigma\eta) - \sinh(\varsigma\eta) - \eta\cosh(\eta) + \sinh(\eta)} - \frac{3}{\eta}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        if nondimensional_force == 0.0 {
            Ok(0.0)
        } else {
            let eta = nondimensional_force;
            let eta_sinh = eta.sinh();
            let varsigma = self.maximum_nondimensional_extension();
            let varsigma_eta = varsigma * eta;
            let varsigma_eta_sinh = varsigma_eta.sinh();
            Ok(eta * (varsigma.powi(2) * varsigma_eta_sinh - eta_sinh)
                / (varsigma_eta * varsigma_eta.cosh() - varsigma_eta_sinh - eta * eta.cosh()
                    + eta_sinh)
                - 3.0 / eta)
        }
    }
    /// ```math
    /// \zeta(\eta) = \frac{\left(\varsigma^2\sinh(\varsigma\eta)+\varsigma^3\eta\cosh(\varsigma\eta)-\sinh(\eta)-\eta\cosh(\eta)\right)\left(\varsigma\eta\cosh(\varsigma\eta)-\sinh(\varsigma\eta)-\eta\cosh(\eta)+\sinh(\eta)\right)-\left(\varsigma^2\eta\sinh(\varsigma\eta)-\eta\sinh(\eta)\right)^2}{\left(\varsigma\eta\cosh(\varsigma\eta)-\sinh(\varsigma\eta)-\eta\cosh(\eta)+\sinh(\eta)\right)^2}+\frac{3}{\eta^2}
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        if nondimensional_force == 0.0 {
            Ok(Scalar::NAN)
        } else {
            let eta = nondimensional_force;
            let eta_sinh = eta.sinh();
            let eta_cosh = eta.cosh();
            let varsigma = self.maximum_nondimensional_extension();
            let varsigma_eta = varsigma * nondimensional_force;
            let varsigma_eta_sinh = varsigma_eta.sinh();
            let varsigma_eta_cosh = varsigma_eta.cosh();
            let a = eta * (varsigma * varsigma * varsigma_eta_sinh - eta_sinh);
            let b =
                varsigma_eta * varsigma_eta_cosh - varsigma_eta_sinh - eta * eta_cosh + eta_sinh;
            let a_prime = (varsigma * varsigma * varsigma_eta_sinh - eta_sinh)
                + eta * (varsigma * varsigma * varsigma * varsigma_eta_cosh - eta_cosh);
            Ok((a_prime * b - a * a) / (b * b) + 3.0 / (eta * eta))
        }
    }
}

impl Legendre for SquareWellFreelyJointedChain {
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl MonteCarlo for SquareWellFreelyJointedChain {
    fn random_configuration(&self) -> Configuration {
        let mut position = CurrentCoordinate::zero();
        let max_strain = self.maximum_nondimensional_extension() - 1.0;
        (0..self.number_of_links())
            .map(|_| {
                let cos_theta = 2.0 * random_uniform() - 1.0;
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let phi = TAU * random_uniform();
                let (sin_phi, cos_phi) = phi.sin_cos();
                let lambda = 1.0 + max_strain * random_uniform();
                position[0] += lambda * sin_theta * cos_phi;
                position[1] += lambda * sin_theta * sin_phi;
                position[2] += lambda * cos_theta;
                position.clone()
            })
            .collect()
    }
}
