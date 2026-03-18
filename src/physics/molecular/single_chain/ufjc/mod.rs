#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar,
        special::{langevin, langevin_derivative, sinhc},
    },
    physics::{
        BOLTZMANN_CONSTANT,
        molecular::{
            potential::Potential,
            single_chain::{
                Ensemble, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
                Thermodynamics,
            },
        },
    },
};

/// The extensible freely-jointed chain model.
#[derive(Clone, Debug)]
pub struct Foo<T>
where
    T: Potential,
{
    /// The link potential $`u`$.
    pub link_potential: T,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl<T> Foo<T>
where
    T: Potential,
{
    pub fn nondimensional_link_stiffness(&self, temperature: Scalar) -> Scalar {
        self.link_potential.stiffness(0.0) * self.link_length().powi(2)
            / BOLTZMANN_CONSTANT
            / temperature
    }
}

impl<T> SingleChain for Foo<T>
where
    T: Potential,
{
    fn link_length(&self) -> Scalar {
        self.link_potential.rest_length()
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl<T> Thermodynamics for Foo<T>
where
    T: Potential,
{
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl<T> Isometric for Foo<T>
where
    T: Potential,
{
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

impl<T> Isotensional for Foo<T>
where
    T: Potential,
{
    /// ```math
    /// \beta\varphi(\eta) = ???
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        //
        // uFJC impl will use this and an enum for potentials
        // then put exact here for EFJC
        // and separate helper functions for the common terms between both
        //
        let eta = nondimensional_force;
        let kappa = self.nondimensional_link_stiffness(temperature);
        Ok(self.number_of_links() as Scalar
            * -((sinhc(eta) * (1.0 + eta / kappa / eta.tanh())).ln() + 0.5 * eta.powi(2) / kappa))
    }
    /// ```math
    /// \gamma(\eta) = ???
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        if nondimensional_force == 0.0 {
            Ok(0.0)
        } else {
            let eta = nondimensional_force;
            let kappa = self.nondimensional_link_stiffness(temperature);
            let eta_coth = 1.0 / eta.tanh();
            let gamma_0 = langevin(eta);
            let c = todo!("need third derivative");
            let delta_lambda = todo!("potentials dont use temperature so hard to use eta");
            todo!()
            // Ok(gamma_0
            //     + delta_lambda
            //         * (1.0 + (1.0 - gamma_0 * eta_coth) / (c + (eta / kappa) * eta_coth)))
        }
    }
    /// ```math
    /// c(\eta) = ???
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let temperature = crate::physics::ROOM_TEMPERATURE;
        let kappa = self.nondimensional_link_stiffness(temperature);
        if nondimensional_force == 0.0 {
            Ok(1.0 / 3.0 + (5.0 / 3.0 * kappa + 1.0) / kappa / (kappa + 1.0))
        } else {
            let eta = nondimensional_force;
            let eta_tanh = eta.tanh();
            let eta_coth = 1.0 / eta_tanh;
            let gamma_0 = langevin(eta);
            let delta_lambda = eta / kappa;
            let c_0 = langevin_derivative(eta);
            let g = 1.0 - gamma_0 * eta_coth;
            let h = 1.0 + delta_lambda * eta_coth;
            let dcth = 1.0 - 1.0 / (eta_tanh * eta_tanh);
            let dg = -(c_0 * eta_coth + gamma_0 * dcth);
            let dh = eta_coth / kappa + delta_lambda * dcth;
            Ok(c_0 + (1.0 + g / h) / kappa + delta_lambda * (dg * h - g * dh) / (h * h))
        }
    }
}

impl<T> Legendre for Foo<T>
where
    T: Potential,
{
    fn nondimensional_spherical_distribution(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        unimplemented!()
    }
}
