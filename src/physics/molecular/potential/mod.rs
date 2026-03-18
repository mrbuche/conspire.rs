#[cfg(test)]
mod test;

mod harmonic;
// mod lennard_jones;
mod morse;

pub use harmonic::Harmonic;
pub use morse::Morse;

use crate::math::Scalar;
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
    /// f(x) = \frac{\partial u}{\partial x}
    /// ```
    fn force(&self, length: Scalar) -> Scalar;
    /// ```math
    /// k(x) = \frac{\partial f}{\partial x}
    /// ```
    fn stiffness(&self, length: Scalar) -> Scalar;
    /// ```math
    /// v(f) = u(x) - f\Delta x
    /// ```
    fn legendre(&self, force: Scalar) -> Scalar {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length) - force * extension
    }
    /// ```math
    /// \Delta x(f) = -\frac{\partial v}{\partial f}
    /// ```
    fn extension(&self, force: Scalar) -> Scalar;
    /// ```math
    /// c(x) = \frac{\partial\Delta x}{\partial f}
    /// ```
    fn compliance(&self, force: Scalar) -> Scalar;
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
