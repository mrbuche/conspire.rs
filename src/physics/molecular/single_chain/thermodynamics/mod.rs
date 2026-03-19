use crate::{
    math::{
        Scalar, Tensor, Vector,
        optimize::{EqualityConstraint, LineSearch, NewtonRaphson, SecondOrderOptimization},
    },
    mechanics::CurrentCoordinates,
    physics::molecular::single_chain::{Inextensible, SingleChain, SingleChainError},
};
use std::{f64::consts::PI, thread};

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
    ) -> Result<Scalar, SingleChainError>;
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
    Self: Inextensible + Sync,
{
    fn nondimensional_radial_distribution<const N: usize>(
        &self,
        num_bins: usize,
        num_samples: usize,
        num_threads: usize,
    ) -> (Vector, Vector) {
        let base = num_samples / num_threads;
        let remainder = num_samples % num_threads;
        let max_extension = self.maximum_nondimensional_extension();
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for t in 0..num_threads {
                let samples_t = base + usize::from(t < remainder);
                handles.push(s.spawn(move || {
                    self.nondimensional_radial_distribution_inner::<N>(num_bins, samples_t)
                }));
            }
            let mut total_counts = vec![0; num_bins];
            for h in handles {
                let counts = h.join().expect("thread done goofed");
                for (tot, c) in total_counts.iter_mut().zip(counts) {
                    *tot += c;
                }
            }
            let bin_width = max_extension / (num_bins as Scalar);
            let bin_centers = (0..num_bins)
                .map(|i| (i as Scalar + 0.5) * bin_width)
                .collect();
            let total_samples = num_samples as Scalar;
            let bin_values = total_counts
                .into_iter()
                .map(|count| count as Scalar / total_samples / bin_width)
                .collect();
            (bin_centers, bin_values)
        })
    }
    fn nondimensional_radial_distribution_inner<const N: usize>(
        &self,
        num_bins: usize,
        num_samples: usize,
    ) -> Vec<usize> {
        let mut bin_counts = vec![0; num_bins];
        let num_links = N as Scalar;
        let max_extension = self.maximum_nondimensional_extension();
        for _ in 0..num_samples {
            let configuration = self.random_configuration::<N>();
            let nondimensional_extension = configuration[N - 1].norm() / num_links;
            let bin_index =
                (nondimensional_extension / max_extension * num_bins as Scalar) as usize;
            bin_counts[bin_index] += 1;
        }
        bin_counts
    }
    fn random_configuration<const N: usize>(&self) -> CurrentCoordinates<N>;
}
