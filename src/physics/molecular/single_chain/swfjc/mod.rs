#[cfg(test)]
mod test;

use crate::{
    math::{Scalar, TensorArray},
    mechanics::{CurrentCoordinate, CurrentCoordinates},
    physics::molecular::single_chain::{
        Ensemble, Inextensible, Isometric, Isotensional, MonteCarlo, SingleChain, Legendre, SingleChainError,
        Thermodynamics,
    },
    random_uniform,
};
use std::f64::consts::TAU;

/// The square-well freely-jointed chain model.
#[derive(Clone, Debug)]
pub struct Foo {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The well width $`w_b`$.
    pub well_width: Scalar,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for Foo {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for Foo {
    /// ```math
    /// \lim_{\eta\to\infty}\gamma(\eta) = 1 + \frac{w_b}{\ell_b} = \varsigma
    /// ```
    fn maximum_nondimensional_extension(&self) -> Scalar {
        1.0 + self.well_width / self.link_length
    }
}

impl Thermodynamics for Foo {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for Foo {
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

impl Isotensional for Foo {
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
            let varsigma = self.maximum_nondimensional_extension();
            let varsigma_eta = varsigma * nondimensional_force;
            let varsigma_eta_sinh = varsigma_eta.sinh();
            let eta_sinh = nondimensional_force.sinh();
            Ok(
                nondimensional_force * (varsigma.powi(2) * varsigma_eta_sinh - eta_sinh)
                    / (varsigma_eta * varsigma_eta.cosh()
                        - varsigma_eta_sinh
                        - nondimensional_force * nondimensional_force.cosh()
                        + eta_sinh)
                    - 3.0 / nondimensional_force,
            )
        }
    }
    /// ```math
    /// c(\eta) = ???
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let a = self.maximum_nondimensional_extension();
        let x = nondimensional_force;
    }
}

impl Legendre for Foo {
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        todo!("can probably get pretty close with Langevin guess and then converge")
    }
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}

impl MonteCarlo for Foo {
    fn random_configuration<const N: usize>(&self) -> CurrentCoordinates<N> {
        let mut position = CurrentCoordinate::zero();
        let max_strain = self.maximum_nondimensional_extension() - 1.0;
        (0..N)
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
