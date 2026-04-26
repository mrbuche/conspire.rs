#[cfg(test)]
mod test;

mod harmonic;
// mod lennard_jones;
mod morse;

pub use harmonic::Harmonic;
pub use morse::Morse;

use crate::{math::Scalar, physics::BOLTZMANN_CONSTANT};
use std::fmt::Debug;

pub trait Potential
where
    Self: Clone + Debug,
{
    /// ```math
    /// u = u(x)
    /// ```
    fn energy(&self, length: Scalar) -> Scalar;
    /// ```math
    /// \beta u = \beta u(\lambda)
    /// ```
    fn nondimensional_energy(&self, nondimensional_length: Scalar, temperature: Scalar) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.energy(length) / BOLTZMANN_CONSTANT / temperature
    }
    /// ```math
    /// u = u(x(f))
    /// ```
    fn energy_at_force(&self, force: Scalar) -> Scalar {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length)
    }
    /// ```math
    /// \beta u = \beta u(\lambda(\eta))
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
    /// \eta(\lambda) = \frac{\partial\beta u}{\partial \lambda}
    /// ```
    fn nondimensional_force(&self, nondimensional_length: Scalar, temperature: Scalar) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.force(length) * self.rest_length() / BOLTZMANN_CONSTANT / temperature
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
    /// \beta v(\eta) = \beta u(\lambda) - \eta\Delta\lambda
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
    /// \Delta \lambda(\eta) = -\frac{\partial\beta v}{\partial\eta}
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
