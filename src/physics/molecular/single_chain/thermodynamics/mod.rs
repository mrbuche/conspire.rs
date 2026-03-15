use crate::{
    math::{
        Scalar,
        optimize::{EqualityConstraint, LineSearch, NewtonRaphson, SecondOrderOptimization},
    },
    physics::molecular::single_chain::{SingleChain, SingleChainError},
};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
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
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => Isometric::nondimensional_helmholtz_free_energy_per_link(
                self,
                nondimensional_extension,
            ),
            Ensemble::Isotensional => Legendre::nondimensional_helmholtz_free_energy_per_link(
                self,
                nondimensional_extension,
            ),
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
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => {
                Isometric::nondimensional_stiffness(self, nondimensional_extension)
            }
            Ensemble::Isotensional => {
                Legendre::nondimensional_stiffness(self, nondimensional_extension)
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
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => {
                Legendre::nondimensional_gibbs_free_energy_per_link(self, nondimensional_force)
            }
            Ensemble::Isotensional => {
                Isotensional::nondimensional_gibbs_free_energy_per_link(self, nondimensional_force)
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
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric => Legendre::nondimensional_compliance(self, nondimensional_force),
            Ensemble::Isotensional => {
                Isotensional::nondimensional_compliance(self, nondimensional_force)
            }
        }
    }
}

pub trait Isometric
where
    Self: SingleChain,
{
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_helmholtz_free_energy(nondimensional_extension)?
                / (self.number_of_links() as Scalar),
        )
    }
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_stiffness(
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

pub trait Isotensional
where
    Self: SingleChain,
{
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(self.nondimensional_gibbs_free_energy(nondimensional_force)?
            / (self.number_of_links() as Scalar))
    }
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    fn nondimensional_compliance(
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
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_helmholtz_free_energy(self, nondimensional_extension)?
                / (self.number_of_links() as Scalar),
        )
    }
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match (NewtonRaphson {
            abs_tol: 1e-10,
            line_search: LineSearch::Error {
                cut_back: 5e-1,
                max_steps: 10,
            },
            ..Default::default()
        }
        .minimize(
            |&nondimensional_force| {
                Ok(Isotensional::nondimensional_gibbs_free_energy_per_link(
                    self,
                    nondimensional_force,
                )? - nondimensional_force * nondimensional_extension)
            },
            |&nondimensional_force| {
                Ok(
                    Isotensional::nondimensional_extension(self, nondimensional_force)?
                        - nondimensional_extension,
                )
            },
            |&nondimensional_force| {
                Ok(Isotensional::nondimensional_compliance(
                    self,
                    nondimensional_force,
                )?)
            },
            nondimensional_extension,
            EqualityConstraint::None,
            None,
        )) {
            Ok(nondimensional_force) => Ok(nondimensional_force),
            Err(error) => Err(SingleChainError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_force = Legendre::nondimensional_force(self, nondimensional_extension)?;
        Ok(1.0 / Isotensional::nondimensional_compliance(self, nondimensional_force)?)
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
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_gibbs_free_energy(self, nondimensional_force)?
                / (self.number_of_links() as Scalar),
        )
    }
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match (NewtonRaphson {
            abs_tol: 1e-10,
            line_search: LineSearch::Error {
                cut_back: 5e-1,
                max_steps: 10,
            },
            ..Default::default()
        }
        .minimize(
            |&nondimensional_extension| {
                Ok(Isometric::nondimensional_helmholtz_free_energy_per_link(
                    self,
                    nondimensional_extension,
                )? - nondimensional_force * nondimensional_extension)
            },
            |&nondimensional_extension| {
                Ok(
                    Isometric::nondimensional_force(self, nondimensional_extension)?
                        - nondimensional_force,
                )
            },
            |&nondimensional_extension| {
                Ok(Isometric::nondimensional_stiffness(
                    self,
                    nondimensional_extension,
                )?)
            },
            nondimensional_force,
            EqualityConstraint::None,
            None,
        )) {
            Ok(nondimensional_extension) => Ok(nondimensional_extension),
            Err(error) => Err(SingleChainError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_extension =
            Legendre::nondimensional_extension(self, nondimensional_force)?;
        Ok(1.0 / Isometric::nondimensional_stiffness(self, nondimensional_extension)?)
    }
}
