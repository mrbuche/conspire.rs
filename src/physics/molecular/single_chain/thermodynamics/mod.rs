use crate::{
    math::Scalar,
    physics::molecular::single_chain::{SingleChain, SingleChainError},
};
use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub enum Ensemble {
    Isometric,
    Isotensional,
}

pub trait Thermodynamics
where
    Self: Isometric + Isotensional + Legendre + SingleChain,
{
    fn ensemble(&self) -> Ensemble;
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => {
                Isometric::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
            Ensemble::Isotensional => {
                Legendre::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => Isometric::nondimensional_force(self, nondimensional_extension),
            Ensemble::Isotensional => {
                Legendre::nondimensional_force(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => {
                Legendre::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
            Ensemble::Isotensional => {
                Isotensional::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
        }
    }
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => Legendre::nondimensional_extension(self, nondimensional_force),
            Ensemble::Isotensional => {
                Isotensional::nondimensional_extension(self, nondimensional_force)
            }
        }
    }
}

pub trait Isometric {
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_spherical_distribution(nondimensional_extension)?
                * (4.0 * PI * nondimensional_extension.powi(2)),
        )
    }
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
}

pub trait Isotensional {
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
}

pub trait Legendre
where
    Self: Isometric + Isotensional + SingleChain,
{
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_force = Legendre::nondimensional_force(self, nondimensional_extension)?;
        Ok(
            Isotensional::nondimensional_gibbs_free_energy(self, nondimensional_force)?
                + self.number_of_links() as Scalar
                    * nondimensional_force
                    * nondimensional_extension,
        )
    }
    fn nondimensional_force(
        &self,
        _nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("default impl invert isotensional nondimensional extension")
    }
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_extension =
            Legendre::nondimensional_extension(self, nondimensional_force)?;
        Ok(
            Isometric::nondimensional_helmholtz_free_energy(self, nondimensional_extension)?
                - self.number_of_links() as Scalar
                    * nondimensional_force
                    * nondimensional_extension,
        )
    }
    fn nondimensional_extension(
        &self,
        _nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        todo!("default impl invert isometric nondimensional force")
    }
}
