#[cfg(test)]
mod test;

mod harmonic;
// mod lennard_jones;
mod morse;

pub use harmonic::Harmonic;
pub use morse::Morse;

use crate::{
    math::{
        Quantity, Scalar,
        unit::{
            Energy, Force, ForcePerLength, Length, ReciprocalForcePerLength, Stress, Temperature,
        },
    },
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
    fn energy(&self, length: Quantity<Length>) -> Quantity<Energy>;
    /// ```math
    /// \upsilon(\lambda) = \beta u
    /// ```
    fn nondimensional_energy(
        &self,
        nondimensional_length: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        self.energy(length).ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// u = u[x(f)]
    /// ```
    fn energy_at_force(&self, force: Quantity<Force>) -> Quantity<Energy> {
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
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let force = BOLTZMANN_CONSTANT * temperature / self.rest_length() * nondimensional_force;
        self.energy_at_force(force)
            .ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// f(x) = \frac{\partial u}{\partial x}
    /// ```
    fn force(&self, length: Quantity<Length>) -> Quantity<Force>;
    /// ```math
    /// \eta(\lambda) = \frac{\partial\upsilon}{\partial \lambda}
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_length: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        (self.force(length) * self.rest_length()).ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// f = x^{-1}[u^{-1}(u)]
    /// ```
    fn forces_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Force>; 2];
    /// ```math
    /// \eta = \lambda^{-1}[\upsilon^{-1}(\upsilon)]
    /// ```
    fn nondimensional_forces_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Quantity<Temperature>,
    ) -> [Scalar; 2] {
        let energy = BOLTZMANN_CONSTANT * temperature * nondimensional_energy;
        self.forces_at_energy(energy)
            .map(|force| (force * self.rest_length()).ratio(BOLTZMANN_CONSTANT * temperature))
    }
    /// ```math
    /// k(x) = \frac{\partial f}{\partial x}
    /// ```
    fn stiffness(&self, length: Quantity<Length>) -> Quantity<ForcePerLength>;
    /// ```math
    /// \kappa(x) = \frac{\partial\eta}{\partial\lambda}
    /// ```
    fn nondimensional_stiffness(
        &self,
        nondimensional_length: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        (self.stiffness(length) * (self.rest_length() * self.rest_length()))
            .ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// h(x) = \frac{\partial k}{\partial x}
    /// ```
    fn anharmonicity(&self, length: Quantity<Length>) -> Quantity<Stress>;
    /// ```math
    /// g(x) = \frac{\partial\kappa}{\partial\lambda}
    /// ```
    fn nondimensional_anharmonicity(
        &self,
        nondimensional_length: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let length = self.rest_length() * nondimensional_length;
        (self.anharmonicity(length)
            * (self.rest_length() * self.rest_length() * self.rest_length()))
        .ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// v(f) = u(x) - f\Delta x
    /// ```
    fn legendre(&self, force: Quantity<Force>) -> Quantity<Energy> {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length) - force * extension
    }
    /// ```math
    /// \nu(\eta) = \upsilon(\lambda) - \eta\Delta\lambda
    /// ```
    fn nondimensional_legendre(
        &self,
        nondimensional_force: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let force = BOLTZMANN_CONSTANT * temperature / self.rest_length() * nondimensional_force;
        self.legendre(force).ratio(BOLTZMANN_CONSTANT * temperature)
    }
    /// ```math
    /// \Delta x(f) = -\frac{\partial v}{\partial f}
    /// ```
    fn extension(&self, force: Quantity<Force>) -> Quantity<Length>;
    /// ```math
    /// \Delta\lambda(\eta) = -\frac{\partial\nu}{\partial\eta}
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let force = BOLTZMANN_CONSTANT * temperature / self.rest_length() * nondimensional_force;
        self.extension(force).ratio(self.rest_length())
    }
    /// ```math
    /// x(f) = x_0 + \Delta x(f)
    /// ```
    fn length(&self, force: Quantity<Force>) -> Quantity<Length> {
        self.rest_length() + self.extension(force)
    }
    /// ```math
    /// \lambda(\eta) = 1 + \Delta\lambda(\eta)
    /// ```
    fn nondimensional_length(
        &self,
        nondimensional_force: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        1.0 + self.nondimensional_extension(nondimensional_force, temperature)
    }
    /// ```math
    /// \Delta x = u^{-1}(u) - x_0
    /// ```
    fn extensions_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Length>; 2];
    /// ```math
    /// \Delta\lambda = \upsilon^{-1}(\upsilon) - 1
    /// ```
    fn nondimensional_extensions_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Quantity<Temperature>,
    ) -> [Scalar; 2] {
        let energy = BOLTZMANN_CONSTANT * temperature * nondimensional_energy;
        self.extensions_at_energy(energy)
            .map(|extension| extension.ratio(self.rest_length()))
    }
    /// ```math
    /// x = u^{-1}(u)
    /// ```
    fn lengths_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Length>; 2] {
        self.extensions_at_energy(energy)
            .map(|extension| extension + self.rest_length())
    }
    /// ```math
    /// \lambda = \upsilon^{-1}(\upsilon)
    /// ```
    fn nondimensional_lengths_at_nondimensional_energy(
        &self,
        nondimensional_energy: Scalar,
        temperature: Quantity<Temperature>,
    ) -> [Scalar; 2] {
        self.nondimensional_extensions_at_nondimensional_energy(nondimensional_energy, temperature)
            .map(|extension| extension + 1.0)
    }
    /// ```math
    /// c(x) = \frac{\partial\Delta x}{\partial f}
    /// ```
    fn compliance(&self, force: Quantity<Force>) -> Quantity<ReciprocalForcePerLength>;
    /// ```math
    /// \zeta(x) = \frac{\partial\Delta\lambda}{\partial\eta}
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
        temperature: Quantity<Temperature>,
    ) -> Scalar {
        let force = BOLTZMANN_CONSTANT * temperature / self.rest_length() * nondimensional_force;
        (self.compliance(force) * (BOLTZMANN_CONSTANT * temperature))
            .ratio(self.rest_length() * self.rest_length())
    }
    /// ```math
    /// \text{arg max }u(x) = x_\mathrm{peak}
    /// ```
    fn peak(&self) -> Quantity<Length>;
    /// ```math
    /// f(x_\mathrm{peak}) = f_\mathrm{peak}
    /// ```
    fn peak_force(&self) -> Quantity<Force>;
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Quantity<Length>;
}
