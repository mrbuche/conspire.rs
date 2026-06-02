use crate::{
    math::{
        Scalar, SquareMatrix, Tensor, TensorArray, Vector,
        optimize::{EqualityConstraint, LineSearch, NewtonRaphson, SecondOrderOptimization},
    },
    mechanics::{Coordinates, CurrentCoordinate},
    physics::molecular::single_chain::{Extensible, Inextensible, SingleChain, SingleChainError},
};
use std::{f64::consts::PI, thread::scope};

pub type Configuration = Coordinates<1>;

#[derive(Clone, Copy, Debug)]
pub enum Ensemble {
    Isometric(Scalar),
    Isotensional(Scalar),
}

pub trait Thermodynamics
where
    Self: Isometric + Isotensional + Legendre + SingleChain,
{
    fn ensemble(&self) -> Ensemble;
    fn temperature(&self) -> Scalar {
        match self.ensemble() {
            Ensemble::Isometric(temperature) => temperature,
            Ensemble::Isotensional(temperature) => temperature,
        }
    }
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Isometric::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
            Ensemble::Isotensional(_) => {
                Legendre::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => Isometric::nondimensional_helmholtz_free_energy_per_link(
                self,
                nondimensional_extension,
            ),
            Ensemble::Isotensional(_) => Legendre::nondimensional_helmholtz_free_energy_per_link(
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
            Ensemble::Isometric(_) => {
                Isometric::nondimensional_force(self, nondimensional_extension)
            }
            Ensemble::Isotensional(_) => {
                Legendre::nondimensional_force(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Isometric::nondimensional_stiffness(self, nondimensional_extension)
            }
            Ensemble::Isotensional(_) => {
                Legendre::nondimensional_stiffness(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Isometric::nondimensional_radial_distribution(self, nondimensional_extension)
            }
            Ensemble::Isotensional(_) => {
                Legendre::nondimensional_radial_distribution(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Isometric::nondimensional_spherical_distribution(self, nondimensional_extension)
            }
            Ensemble::Isotensional(_) => {
                Legendre::nondimensional_spherical_distribution(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Legendre::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
            Ensemble::Isotensional(_) => {
                Isotensional::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
        }
    }
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Legendre::nondimensional_gibbs_free_energy_per_link(self, nondimensional_force)
            }
            Ensemble::Isotensional(_) => {
                Isotensional::nondimensional_gibbs_free_energy_per_link(self, nondimensional_force)
            }
        }
    }
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Legendre::nondimensional_extension(self, nondimensional_force)
            }
            Ensemble::Isotensional(_) => {
                Isotensional::nondimensional_extension(self, nondimensional_force)
            }
        }
    }
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                Legendre::nondimensional_compliance(self, nondimensional_force)
            }
            Ensemble::Isotensional(_) => {
                Isotensional::nondimensional_compliance(self, nondimensional_force)
            }
        }
    }
}

pub trait ThermodynamicsExtensible
where
    Self: IsotensionalExtensible + Thermodynamics,
{
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_energy_average(
                    self,
                    nondimensional_force,
                )
            }
        }
    }
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_energy_variance(
                    self,
                    nondimensional_force,
                )
            }
        }
    }
    fn nondimensional_link_energy_probability(
        &self,
        nondimensional_energy: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_energy_probability(
                    self,
                    nondimensional_energy,
                    nondimensional_force,
                )
            }
        }
    }
    fn nondimensional_link_length_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_length_average(
                    self,
                    nondimensional_force,
                )
            }
        }
    }
    fn nondimensional_link_length_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_length_variance(
                    self,
                    nondimensional_force,
                )
            }
        }
    }
    fn nondimensional_link_length_probability(
        &self,
        nondimensional_length: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        match self.ensemble() {
            Ensemble::Isometric(_) => {
                unimplemented!()
            }
            Ensemble::Isotensional(_) => {
                IsotensionalExtensible::nondimensional_link_length_probability(
                    self,
                    nondimensional_length,
                    nondimensional_force,
                )
            }
        }
    }
}

pub trait Isometric
where
    Self: SingleChain,
{
    /// ```math
    /// \beta\psi(\gamma) = -\ln Q(\gamma)
    /// ```
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_helmholtz_free_energy_per_link(nondimensional_extension)?
                * (self.number_of_links() as Scalar),
        )
    }
    /// ```math
    /// \vartheta(\gamma) = \beta\psi(\gamma) / N_b
    /// ```
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_helmholtz_free_energy(nondimensional_extension)?
                / (self.number_of_links() as Scalar),
        )
    }
    /// ```math
    /// \eta(\gamma) = \frac{\partial\vartheta}{\partial\gamma}
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// k(\gamma) = \frac{\partial\eta}{\partial\gamma}
    /// ```
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// \mathcal{g}(\gamma) = 4\pi\gamma^2\mathcal{P}(\gamma)
    /// ```
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_spherical_distribution(nondimensional_extension)?
                * (4.0 * PI * nondimensional_extension.powi(2)),
        )
    }
    /// ```math
    /// \mathcal{P}(\gamma) \propto e^{-\beta\psi(\gamma)}
    /// ```
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError>;
}

pub trait Isotensional
where
    Self: SingleChain,
{
    /// ```math
    /// \beta\varphi(\eta) = -\ln Z(\eta)
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            self.nondimensional_gibbs_free_energy_per_link(nondimensional_force)?
                * (self.number_of_links() as Scalar),
        )
    }
    /// ```math
    /// \varrho(\eta) = \beta\varphi(\eta) / N_b
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(self.nondimensional_gibbs_free_energy(nondimensional_force)?
            / (self.number_of_links() as Scalar))
    }
    /// ```math
    /// \gamma(\eta) = -\frac{\partial\varrho}{\partial\eta}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// c(\eta) = \frac{\partial\gamma}{\partial\eta}
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
}

pub trait IsotensionalExtensible
where
    Self: Extensible + Isotensional,
{
    /// ```math
    /// \langle\upsilon\rangle = \varepsilon\,\frac{\partial\varrho}{\partial\varepsilon}
    /// ```
    fn nondimensional_link_energy_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// \sigma_\upsilon^2 = -\varepsilon^2\frac{\partial^2\varrho}{\partial\varepsilon^2}
    /// ```
    fn nondimensional_link_energy_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// p(\upsilon\,|\,\eta) = \int p(\lambda\,|\,\eta)\,\delta[\upsilon - \upsilon(\lambda)]\,d\lambda
    /// ```
    fn nondimensional_link_energy_probability(
        &self,
        nondimensional_energy: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// \langle\lambda\rangle = \int_0^\infty p(\lambda\,|\,\eta)\,\lambda\,d\lambda
    /// ```
    fn nondimensional_link_length_average(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// \sigma_\lambda^2 = \langle\lambda^2\rangle - \langle\lambda\rangle^2
    /// ```
    fn nondimensional_link_length_variance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
    /// ```math
    /// p(\lambda\,|\,\eta) = \frac{z_0(\eta,\lambda)}{z(\eta)}\,e^{-\upsilon(\lambda)}
    /// ```
    fn nondimensional_link_length_probability(
        &self,
        nondimensional_length: Scalar,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError>;
}

pub trait Legendre
where
    Self: Isometric + Isotensional + SingleChain,
{
    /// ```math
    /// \beta\psi(\gamma) = \beta\varphi(\eta) + N_b\eta(\gamma)\gamma
    /// ```
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
    /// ```math
    /// \vartheta(\gamma) = \varrho(\eta) + \eta(\gamma)\gamma
    /// ```
    fn nondimensional_helmholtz_free_energy_per_link(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_helmholtz_free_energy(self, nondimensional_extension)?
                / (self.number_of_links() as Scalar),
        )
    }
    /// ```math
    /// \eta(\gamma) = \gamma^{-1}(\gamma)
    /// ```
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
    /// ```math
    /// k(\gamma) = \left(\frac{\partial\gamma}{\partial\eta}\right)^{-1}
    /// ```
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_force = Legendre::nondimensional_force(self, nondimensional_extension)?;
        Ok(1.0 / Isotensional::nondimensional_compliance(self, nondimensional_force)?)
    }
    /// ```math
    /// \mathcal{g}(\gamma) = 4\pi\gamma^2\mathcal{P}(\gamma)
    /// ```
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_spherical_distribution(self, nondimensional_extension)?
                * (4.0 * PI * nondimensional_extension.powi(2)),
        )
    }
    /// ```math
    /// \mathcal{P}(\gamma) \propto e^{-\beta\psi(\gamma)}
    /// ```
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_radial_distribution(self, nondimensional_extension)?
                / (4.0 * PI * nondimensional_extension.powi(2)),
        )
    }
    /// ```math
    /// \beta\varphi(\eta) = \beta\psi(\gamma) - N_b\eta\gamma(\eta)
    /// ```
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
    /// ```math
    /// \varrho(\eta) = \vartheta(\gamma) - \eta\gamma(\eta)
    /// ```
    fn nondimensional_gibbs_free_energy_per_link(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(
            Legendre::nondimensional_gibbs_free_energy(self, nondimensional_force)?
                / (self.number_of_links() as Scalar),
        )
    }
    /// ```math
    /// \gamma(\eta) = \eta^{-1}(\eta)
    /// ```
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
    /// ```math
    /// c(\eta) = \left(\frac{\partial\eta}{\partial\gamma}\right)^{-1}
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_extension =
            Legendre::nondimensional_extension(self, nondimensional_force)?;
        Ok(1.0 / Isometric::nondimensional_stiffness(self, nondimensional_extension)?)
    }
}

pub trait MonteCarlo
where
    Self: SingleChain + Sync,
{
    fn cosine_moments(
        &self,
        nondimensional_force: Scalar,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, SquareMatrix, Vector, SquareMatrix) {
        cosine_moments_reweighted(
            self,
            nondimensional_force,
            number_of_samples,
            number_of_threads,
        )
    }
    fn nondimensional_longitudinal_extension(
        &self,
        nondimensional_force: Scalar,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> Scalar {
        //
        // Does not work well above a certain force (see EFRC for bias method).
        //
        // Should set up to use random_nondimensional_link_vectors when implemented to work for a given force,
        // and use the re-weighting (possibly with the above bias method) when unimplemented under force.
        //
        nondimensional_longitudinal_extension_reweighted(
            self,
            nondimensional_force,
            number_of_samples,
            number_of_threads,
        )
    }
    fn nondimensional_longitudinal_extension_biased_stretch(
        &self,
        nondimensional_force: Scalar,
        nondimensional_stretch_bias: Scalar,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> Scalar {
        nondimensional_longitudinal_extension_biased_reweighted(
            self,
            nondimensional_force,
            nondimensional_stretch_bias,
            number_of_samples,
            number_of_threads,
        )
    }
    fn random_nondimensional_link_vectors(&self, nondimensional_force: Scalar) -> Configuration;
    fn random_nondimensional_link_vectors_biased_stretch(
        &self,
        _nondimensional_stretch_bias: Scalar,
    ) -> Configuration {
        self.random_nondimensional_link_vectors(0.0)
    }
    fn random_configuration(&self, nondimensional_force: Scalar) -> Configuration {
        let mut position = CurrentCoordinate::zero();
        self.random_nondimensional_link_vectors(nondimensional_force)
            .into_iter()
            .map(|displacement| {
                position += displacement;
                position.clone()
            })
            .collect()
    }
}

pub trait MonteCarloExtensible
where
    Self: Extensible + MonteCarlo,
{
    fn nondimensional_lateral_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
        maximum_nondimensional_extension: Scalar,
    ) -> (Vector, Vector) {
        nondimensional_lateral_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            maximum_nondimensional_extension,
        )
    }
    fn nondimensional_longitudinal_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
        maximum_nondimensional_extension: Scalar,
    ) -> (Vector, Vector) {
        nondimensional_longitudinal_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            maximum_nondimensional_extension,
        )
    }
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
        maximum_nondimensional_extension: Scalar,
    ) -> (Vector, Vector) {
        nondimensional_radial_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            maximum_nondimensional_extension,
        )
    }
    fn nondimensional_transverse_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
        maximum_nondimensional_extension: Scalar,
    ) -> (Vector, Vector) {
        nondimensional_transverse_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            maximum_nondimensional_extension,
        )
    }
}

impl<T> MonteCarloExtensible for T where T: Extensible + MonteCarlo {}

pub trait MonteCarloInextensible
where
    Self: Inextensible + MonteCarlo,
{
    fn nondimensional_angular_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, Vector) {
        nondimensional_angular_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            self.maximum_nondimensional_extension(),
        )
    }
    fn nondimensional_lateral_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, Vector) {
        nondimensional_lateral_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            self.maximum_nondimensional_extension(),
        )
    }
    fn nondimensional_longitudinal_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, Vector) {
        nondimensional_longitudinal_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            self.maximum_nondimensional_extension(),
        )
    }
    fn nondimensional_radial_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, Vector) {
        nondimensional_radial_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            self.maximum_nondimensional_extension(),
        )
    }
    fn nondimensional_transverse_distribution(
        &self,
        nondimensional_force: Scalar,
        num_bins: usize,
        number_of_samples: usize,
        number_of_threads: usize,
    ) -> (Vector, Vector) {
        nondimensional_transverse_distribution(
            self,
            nondimensional_force,
            num_bins,
            number_of_samples,
            number_of_threads,
            self.maximum_nondimensional_extension(),
        )
    }
}

impl<T> MonteCarloInextensible for T where T: Inextensible + MonteCarlo {}

pub fn cosine_moments_reweighted<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_samples: usize,
    number_of_threads: usize,
) -> (Vector, SquareMatrix, Vector, SquareMatrix) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    cosine_moments_reweighted_inner(
                        model,
                        nondimensional_force,
                        base + usize::from(t < remainder),
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .reduce(|mut acc, (x_max, z_scaled, cos1_scaled, cos1cos1_scaled, cos2_scaled, cos2cos1_scaled)| {
                let x_max_new = acc.0.max(x_max);
                let scale_acc = (acc.0 - x_max_new).exp();
                let scale_new = (x_max - x_max_new).exp();
                acc.1 = acc.1 * scale_acc + z_scaled * scale_new;
                acc.2 = acc.2 * scale_acc + cos1_scaled * scale_new;
                acc.3 = acc.3 * scale_acc + cos1cos1_scaled * scale_new;
                acc.4 = acc.4 * scale_acc + cos2_scaled * scale_new;
                acc.5 = acc.5 * scale_acc + cos2cos1_scaled * scale_new;
                acc.0 = x_max_new;
                acc
            })
            .map(|(_x_max, z_scaled, cos1_scaled, cos1cos1_scaled, cos2_scaled, cos2cos1_scaled)| {
                let z_inv = 1.0 / z_scaled;
                (
                    cos1_scaled * z_inv,
                    cos1cos1_scaled * z_inv,
                    cos2_scaled * z_inv,
                    cos2cos1_scaled * z_inv,
                )
            })
            .unwrap()
    })
}

fn cosine_moments_reweighted_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_samples: usize,
) -> (Scalar, Scalar, Vector, SquareMatrix, Vector, SquareMatrix) {
    let num_links = model.number_of_links() as usize;

    let mut x_max = Scalar::NEG_INFINITY;
    let mut z_scaled = 0.0;
    let mut cos1_scaled = Vector::zero(num_links);
    let mut cos1cos1_scaled = SquareMatrix::zero(num_links);
    let mut cos2_scaled = Vector::zero(num_links);
    let mut cos2cos1_scaled = SquareMatrix::zero(num_links);

    for _ in 0..number_of_samples {
        let links = model.random_nondimensional_link_vectors(0.0);

        let cosines: Vec<Scalar> = links.iter().map(|link| link[2] / link.norm()).collect();

        let sum_cos: Scalar = cosines.iter().sum();
        let x = nondimensional_force * sum_cos;

        if x > x_max {
            let scale = if x_max.is_finite() {
                (x_max - x).exp()
            } else {
                0.0
            };
            z_scaled *= scale;
            cos1_scaled *= scale;
            cos1cos1_scaled *= scale;
            cos2_scaled *= scale;
            cos2cos1_scaled *= scale;
            x_max = x;
        }

        let w = (x - x_max).exp();

        z_scaled += w;

        for i in 0..num_links {
            let ci = cosines[i];
            let ci2 = ci * ci;

            cos1_scaled[i] += ci * w;
            cos2_scaled[i] += ci2 * w;

            for j in 0..num_links {
                let cj = cosines[j];
                cos1cos1_scaled[i][j] += ci * cj * w;
                cos2cos1_scaled[i][j] += ci2 * cj * w;
            }
        }
    }
    (
        x_max,
        z_scaled,
        cos1_scaled,
        cos1cos1_scaled,
        cos2_scaled,
        cos2cos1_scaled,
    )
}

fn nondimensional_longitudinal_extension_reweighted<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_samples: usize,
    number_of_threads: usize,
) -> Scalar {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;

    scope(|s| {
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_longitudinal_extension_reweighted_inner(
                        model,
                        nondimensional_force,
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

fn nondimensional_longitudinal_extension_reweighted_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_samples: usize,
) -> (Scalar, Scalar, Scalar) {
    let mut x_max = Scalar::NEG_INFINITY;
    let mut z_scaled = 0.0;
    let mut ext_scaled = 0.0;

    for _ in 0..number_of_samples {
        let links = model.random_nondimensional_link_vectors(0.0);

        let extension_sum: Scalar = links.iter().map(|link| link[2]).sum();
        let x = nondimensional_force * extension_sum;

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

fn nondimensional_longitudinal_extension_biased_reweighted<T: MonteCarlo>(
    model: &T,
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
                    nondimensional_longitudinal_extension_biased_reweighted_inner(
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

fn nondimensional_longitudinal_extension_biased_reweighted_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    nondimensional_stretch_bias: Scalar,
    number_of_samples: usize,
) -> (Scalar, Scalar, Scalar) {
    let mut x_max = Scalar::NEG_INFINITY;
    let mut z_scaled = 0.0;
    let mut ext_scaled = 0.0;
    for _ in 0..number_of_samples {
        let links =
            model.random_nondimensional_link_vectors_biased_stretch(nondimensional_stretch_bias);
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

fn nondimensional_angular_distribution<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_bins: usize,
    number_of_samples: usize,
    number_of_threads: usize,
    maximum_nondimensional_extension: Scalar,
) -> (Vector, Vector) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        let mut total_counts = vec![0; number_of_bins];
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_angular_distribution_inner(
                        model,
                        nondimensional_force,
                        number_of_bins,
                        base + usize::from(t < remainder),
                        maximum_nondimensional_extension,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|handle| {
                total_counts
                    .iter_mut()
                    .zip(handle.join().unwrap())
                    .for_each(|(tot, c)| *tot += c)
            });
        let bin_width = 2.0 * maximum_nondimensional_extension / (number_of_bins as Scalar);
        let bin_centers = (0..number_of_bins)
            .map(|i| -maximum_nondimensional_extension + (i as Scalar + 0.5) * bin_width)
            .collect();
        let total_samples = number_of_samples as Scalar;
        let bin_values = total_counts
            .into_iter()
            .map(|count| count as Scalar / total_samples / bin_width)
            .collect();
        (bin_centers, bin_values)
    })
}

fn nondimensional_lateral_distribution<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_bins: usize,
    number_of_samples: usize,
    number_of_threads: usize,
    maximum_nondimensional_extension: Scalar,
) -> (Vector, Vector) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        let mut total_counts = vec![0; number_of_bins];
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_lateral_distribution_inner(
                        model,
                        nondimensional_force,
                        number_of_bins,
                        base + usize::from(t < remainder),
                        maximum_nondimensional_extension,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|handle| {
                total_counts
                    .iter_mut()
                    .zip(handle.join().unwrap())
                    .for_each(|(tot, c)| *tot += c)
            });
        let bin_width = 2.0 * maximum_nondimensional_extension / (number_of_bins as Scalar);
        let bin_centers = (0..number_of_bins)
            .map(|i| -maximum_nondimensional_extension + (i as Scalar + 0.5) * bin_width)
            .collect();
        let total_samples = number_of_samples as Scalar;
        let bin_values = total_counts
            .into_iter()
            .map(|count| count as Scalar / total_samples / bin_width)
            .collect();
        (bin_centers, bin_values)
    })
}

fn nondimensional_longitudinal_distribution<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_bins: usize,
    number_of_samples: usize,
    number_of_threads: usize,
    maximum_nondimensional_extension: Scalar,
) -> (Vector, Vector) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        let mut total_counts = vec![0; number_of_bins];
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_longitudinal_distribution_inner(
                        model,
                        nondimensional_force,
                        number_of_bins,
                        base + usize::from(t < remainder),
                        maximum_nondimensional_extension,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|handle| {
                total_counts
                    .iter_mut()
                    .zip(handle.join().unwrap())
                    .for_each(|(tot, c)| *tot += c)
            });
        let bin_width = 2.0 * maximum_nondimensional_extension / (number_of_bins as Scalar);
        let bin_centers = (0..number_of_bins)
            .map(|i| -maximum_nondimensional_extension + (i as Scalar + 0.5) * bin_width)
            .collect();
        let total_samples = number_of_samples as Scalar;
        let bin_values = total_counts
            .into_iter()
            .map(|count| count as Scalar / total_samples / bin_width)
            .collect();
        (bin_centers, bin_values)
    })
}

fn nondimensional_radial_distribution<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_bins: usize,
    number_of_samples: usize,
    number_of_threads: usize,
    maximum_nondimensional_extension: Scalar,
) -> (Vector, Vector) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        let mut total_counts = vec![0; number_of_bins];
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_radial_distribution_inner(
                        model,
                        nondimensional_force,
                        number_of_bins,
                        base + usize::from(t < remainder),
                        maximum_nondimensional_extension,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|handle| {
                total_counts
                    .iter_mut()
                    .zip(handle.join().unwrap())
                    .for_each(|(tot, c)| *tot += c)
            });
        let bin_width = maximum_nondimensional_extension / (number_of_bins as Scalar);
        let bin_centers = (0..number_of_bins)
            .map(|i| (i as Scalar + 0.5) * bin_width)
            .collect();
        let total_samples = number_of_samples as Scalar;
        let bin_values = total_counts
            .into_iter()
            .map(|count| count as Scalar / total_samples / bin_width)
            .collect();
        (bin_centers, bin_values)
    })
}

fn nondimensional_transverse_distribution<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    number_of_bins: usize,
    number_of_samples: usize,
    number_of_threads: usize,
    maximum_nondimensional_extension: Scalar,
) -> (Vector, Vector) {
    let base = number_of_samples / number_of_threads;
    let remainder = number_of_samples % number_of_threads;
    scope(|s| {
        let mut total_counts = vec![0; number_of_bins];
        (0..number_of_threads)
            .map(|t| {
                s.spawn(move || {
                    nondimensional_transverse_distribution_inner(
                        model,
                        nondimensional_force,
                        number_of_bins,
                        base + usize::from(t < remainder),
                        maximum_nondimensional_extension,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|handle| {
                total_counts
                    .iter_mut()
                    .zip(handle.join().unwrap())
                    .for_each(|(tot, c)| *tot += c)
            });
        let bin_width = maximum_nondimensional_extension / (number_of_bins as Scalar);
        let bin_centers = (0..number_of_bins)
            .map(|i| (i as Scalar + 0.5) * bin_width)
            .collect();
        let total_samples = number_of_samples as Scalar;
        let bin_values = total_counts
            .into_iter()
            .map(|count| count as Scalar / total_samples / bin_width)
            .collect();
        (bin_centers, bin_values)
    })
}

fn nondimensional_angular_distribution_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    num_bins: usize,
    number_of_samples: usize,
    maximum_nondimensional_extension: Scalar,
) -> Vec<usize> {
    let mut bin_counts = vec![0; num_bins];
    let end_index = model.number_of_links() as usize - 1;
    for _ in 0..number_of_samples {
        let configuration = model.random_configuration(nondimensional_force);
        let gamma = configuration[end_index].norm();
        let nondimensional_extension = if gamma == 0.0 {
            0.0
        } else {
            configuration[end_index][2] / gamma
        };
        if nondimensional_extension.abs() > maximum_nondimensional_extension {
            panic!(
                "Sample {nondimensional_extension} outside [-{maximum_nondimensional_extension}, {maximum_nondimensional_extension}]"
            )
        }
        let bin_index = ((nondimensional_extension + maximum_nondimensional_extension)
            / (2.0 * maximum_nondimensional_extension)
            * num_bins as Scalar) as usize;
        bin_counts[bin_index] += 1;
    }
    bin_counts
}

fn nondimensional_lateral_distribution_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    num_bins: usize,
    number_of_samples: usize,
    maximum_nondimensional_extension: Scalar,
) -> Vec<usize> {
    let mut bin_counts = vec![0; num_bins];
    let num_links = model.number_of_links() as Scalar;
    let end_index = model.number_of_links() as usize - 1;
    for _ in 0..number_of_samples {
        let configuration = model.random_configuration(nondimensional_force);
        let nondimensional_extension = configuration[end_index][1] / num_links;
        if nondimensional_extension.abs() > maximum_nondimensional_extension {
            panic!(
                "Sample {nondimensional_extension} outside [-{maximum_nondimensional_extension}, {maximum_nondimensional_extension}]"
            )
        }
        let bin_index = ((nondimensional_extension + maximum_nondimensional_extension)
            / (2.0 * maximum_nondimensional_extension)
            * num_bins as Scalar) as usize;
        bin_counts[bin_index] += 1;
    }
    bin_counts
}

fn nondimensional_longitudinal_distribution_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    num_bins: usize,
    number_of_samples: usize,
    maximum_nondimensional_extension: Scalar,
) -> Vec<usize> {
    let mut bin_counts = vec![0; num_bins];
    let num_links = model.number_of_links() as Scalar;
    let end_index = model.number_of_links() as usize - 1;
    for _ in 0..number_of_samples {
        let configuration = model.random_configuration(nondimensional_force);
        let nondimensional_extension = configuration[end_index][2] / num_links;
        if nondimensional_extension.abs() > maximum_nondimensional_extension {
            panic!(
                "Sample {nondimensional_extension} outside [-{maximum_nondimensional_extension}, {maximum_nondimensional_extension}]"
            )
        }
        let bin_index = ((nondimensional_extension + maximum_nondimensional_extension)
            / (2.0 * maximum_nondimensional_extension)
            * num_bins as Scalar) as usize;
        bin_counts[bin_index] += 1;
    }
    bin_counts
}

fn nondimensional_radial_distribution_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    num_bins: usize,
    number_of_samples: usize,
    maximum_nondimensional_extension: Scalar,
) -> Vec<usize> {
    let mut bin_counts = vec![0; num_bins];
    let num_links = model.number_of_links() as Scalar;
    let end_index = model.number_of_links() as usize - 1;
    for _ in 0..number_of_samples {
        let configuration = model.random_configuration(nondimensional_force);
        let nondimensional_extension = configuration[end_index].norm() / num_links;
        if nondimensional_extension > maximum_nondimensional_extension {
            panic!(
                "Sample {nondimensional_extension} above maximum {maximum_nondimensional_extension}"
            )
        }
        let bin_index = (nondimensional_extension / maximum_nondimensional_extension
            * num_bins as Scalar) as usize;
        bin_counts[bin_index] += 1;
    }
    bin_counts
}

fn nondimensional_transverse_distribution_inner<T: MonteCarlo>(
    model: &T,
    nondimensional_force: Scalar,
    num_bins: usize,
    number_of_samples: usize,
    maximum_nondimensional_extension: Scalar,
) -> Vec<usize> {
    let mut bin_counts = vec![0; num_bins];
    let num_links = model.number_of_links() as Scalar;
    let end_index = model.number_of_links() as usize - 1;
    for _ in 0..number_of_samples {
        let configuration = model.random_configuration(nondimensional_force);
        let nondimensional_extension =
            (configuration[end_index][0].powi(2) + configuration[end_index][1].powi(2)).sqrt()
                / num_links;
        if nondimensional_extension.abs() > maximum_nondimensional_extension {
            panic!(
                "Sample {nondimensional_extension} outside [-{maximum_nondimensional_extension}, {maximum_nondimensional_extension}]"
            )
        }
        let bin_index = (nondimensional_extension / maximum_nondimensional_extension
            * num_bins as Scalar) as usize;
        bin_counts[bin_index] += 1;
    }
    bin_counts
}
