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
    fn energy(&self, length: Scalar) -> Scalar;
    fn force(&self, length: Scalar) -> Scalar;
    fn stiffness(&self, length: Scalar) -> Scalar;
    fn legendre(&self, force: Scalar) -> Scalar {
        let extension = self.extension(force);
        let length = self.rest_length() + extension;
        self.energy(length) - force * extension
    }
    fn extension(&self, force: Scalar) -> Scalar;
    fn compliance(&self, force: Scalar) -> Scalar;
    fn maximum_force(&self) -> Scalar;
    fn peak(&self) -> Scalar;
    fn rest_length(&self) -> Scalar;
}
