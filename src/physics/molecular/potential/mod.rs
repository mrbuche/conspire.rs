#[cfg(test)]
mod test;

mod harmonic;
// mod lennard_jones;
mod morse;

pub use harmonic::Harmonic;
pub use morse::Morse;

use crate::{
    math::{Scalar, ScalarList},
    physics::BOLTZMANN_CONSTANT,
};
use std::fmt::Debug;

/// Potential models.
pub trait Potential
where
    Self: Clone + Debug,
{
    /// ```math
    /// u = u(x)
    /// ```
    fn energy(&self, length: Scalar) -> Scalar;
    /// ```math
    /// \upsilon(\lambda) = \beta u
    /// ```
    fn nondimensional_energy(&self, nondimensional_length: Scalar, temperature: Scalar) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.energy(length) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// u = u[x(f)]
    /// ```
    fn energy_at_force(&self, force: Scalar) -> Scalar {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length)
    }
    /// ```math
    /// \upsilon = \upsilon[\lambda(\eta)]
    /// ```
    fn nondimensional_energy_at_nondimensional_force(
        &self,
        nondimensional_force: Scalar,
        temperature: Scalar,
    ) -> Scalar {
        let force = nondimensional_force / self.rest_length() * BOLTZMANN_CONSTANT * temperature;
        self.energy_at_force(force) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// f(x) = \frac{\partial u}{\partial x}
    /// ```
    fn force(&self, length: Scalar) -> Scalar;
    /// ```math
    /// \eta(\lambda) = \frac{\partial\upsilon}{\partial \lambda}
    /// ```
    fn nondimensional_force(&self, nondimensional_length: Scalar, temperature: Scalar) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.force(length) * self.rest_length() / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// f = x^{-1}[u^{-1}(u)]
    /// ```
    fn forces_at_energy(&self, energy: Scalar) -> ScalarList<2>;
    /// ```math
    /// \eta = \lambda^{-1}[\upsilon^{-1}(\upsilon)]
    /// ```
    fn nondimensional_forces_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Scalar,
    ) -> ScalarList<2> {
        let energy = nondimensional_energy * BOLTZMANN_CONSTANT * temperature;
        self.forces_at_energy(energy) * self.rest_length() / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// k(x) = \frac{\partial f}{\partial x}
    /// ```
    fn stiffness(&self, length: Scalar) -> Scalar;
    /// ```math
    /// \kappa(x) = \frac{\partial\eta}{\partial\lambda}
    /// ```
    fn nondimensional_stiffness(
        &self,
        nondimensional_length: Scalar,
        temperature: Scalar,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.stiffness(length) * self.rest_length().powi(2) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// h(x) = \frac{\partial k}{\partial x}
    /// ```
    fn anharmonicity(&self, length: Scalar) -> Scalar;
    /// ```math
    /// g(x) = \frac{\partial\kappa}{\partial\lambda}
    /// ```
    fn nondimensional_anharmonicity(
        &self,
        nondimensional_length: Scalar,
        temperature: Scalar,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.anharmonicity(length) * self.rest_length().powi(3) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// v(f) = u(x) - f\Delta x
    /// ```
    fn legendre(&self, force: Scalar) -> Scalar {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length) - force * extension
    }
    /// ```math
    /// \nu(\eta) = \upsilon(\lambda) - \eta\Delta\lambda
    /// ```
    fn nondimensional_legendre(&self, nondimensional_force: Scalar, temperature: Scalar) -> Scalar {
        let force = nondimensional_force / self.rest_length() * BOLTZMANN_CONSTANT * temperature;
        self.legendre(force) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// \Delta x(f) = -\frac{\partial v}{\partial f}
    /// ```
    fn extension(&self, force: Scalar) -> Scalar;
    /// ```math
    /// \Delta\lambda(\eta) = -\frac{\partial\nu}{\partial\eta}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
        temperature: Scalar,
    ) -> Scalar {
        let force = nondimensional_force / self.rest_length() * BOLTZMANN_CONSTANT * temperature;
        self.extension(force) / self.rest_length()
    }
    /// ```math
    /// x(f) = x_0 + \Delta x(f)
    /// ```
    fn length(&self, force: Scalar) -> Scalar {
        self.rest_length() + self.extension(force)
    }
    /// ```math
    /// \lambda(\eta) = 1 + \Delta\lambda(\eta)
    /// ```
    fn nondimensional_length(&self, nondimensional_force: Scalar, temperature: Scalar) -> Scalar {
        1.0 + self.nondimensional_extension(nondimensional_force, temperature)
    }
    /// ```math
    /// \Delta x = u^{-1}(u) - x_0
    /// ```
    fn extensions_at_energy(&self, energy: Scalar) -> ScalarList<2>;
    /// ```math
    /// \Delta\lambda = \upsilon^{-1}(\upsilon) - 1
    /// ```
    fn nondimensional_extensions_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Scalar,
    ) -> ScalarList<2> {
        let energy = nondimensional_energy * BOLTZMANN_CONSTANT * temperature;
        self.extensions_at_energy(energy) / self.rest_length()
    }
    /// ```math
    /// x = u^{-1}(u)
    /// ```
    fn lengths_at_energy(&self, energy: Scalar) -> ScalarList<2> {
        self.extensions_at_energy(energy) + ScalarList::from([self.rest_length(); 2])
    }
    /// ```math
    /// \lambda = \upsilon^{-1}(\upsilon)
    /// ```
    fn nondimensional_lengths_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Scalar,
    ) -> ScalarList<2> {
        self.nondimensional_extensions_at_nondimensional_energy(nondimensional_energy, temperature)
            + ScalarList::from([1.0; 2])
    }
    /// ```math
    /// c(x) = \frac{\partial\Delta x}{\partial f}
    /// ```
    fn compliance(&self, force: Scalar) -> Scalar;
    /// ```math
    /// \zeta(x) = \frac{\partial\Delta\lambda}{\partial\eta}
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
        temperature: Scalar,
    ) -> Scalar {
        let force = nondimensional_force / self.rest_length() * BOLTZMANN_CONSTANT * temperature;
        self.compliance(force) / self.rest_length().powi(2) * BOLTZMANN_CONSTANT * temperature
    }
    /// ```math
    /// \text{arg max }u(x) = x_\mathrm{peak}
    /// ```
    fn peak(&self) -> Scalar;
    /// ```math
    /// f(x_\mathrm{peak}) = f_\mathrm{peak}
    /// ```
    fn peak_force(&self) -> Scalar;
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Scalar;
}
