use crate::{
    math::{
        Quantity, Scalar,
        unit::{Energy, Force, ForcePerLength, Length, ReciprocalForcePerLength, Stress},
    },
    physics::molecular::potential::Potential,
};

/// The harmonic potential.
#[derive(Clone, Debug)]
pub struct Harmonic {
    /// The rest length $`x_0`$.
    pub rest_length: Quantity<Length>,
    /// The stiffness $`k`$.
    pub stiffness: Quantity<ForcePerLength>,
}

impl Potential for Harmonic {
    /// ```math
    /// u(x) = \frac{1}{2}\,k(x - x_0)^2
    /// ```
    fn energy(&self, length: Quantity<Length>) -> Quantity<Energy> {
        let delta = length - self.rest_length;
        0.5 * self.stiffness * (delta * delta)
    }
    /// ```math
    /// f(x) = k(x - x_0)
    /// ```
    fn force(&self, length: Quantity<Length>) -> Quantity<Force> {
        self.stiffness * (length - self.rest_length)
    }
    /// ```math
    /// f = \pm\sqrt{2ku}
    /// ```
    fn forces_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Force>; 2] {
        let force = Quantity::<Force>::new((2.0 * self.stiffness.value() * energy.value()).sqrt());
        [force, -force]
    }
    /// ```math
    /// k(x) = k
    /// ```
    fn stiffness(&self, _length: Quantity<Length>) -> Quantity<ForcePerLength> {
        self.stiffness
    }
    /// ```math
    /// h(x) = 0.0
    /// ```
    fn anharmonicity(&self, _length: Quantity<Length>) -> Quantity<Stress> {
        Quantity::new(0.0)
    }
    /// ```math
    /// \Delta x(f) = \frac{f}{k}
    /// ```
    fn extension(&self, force: Quantity<Force>) -> Quantity<Length> {
        force / self.stiffness
    }
    /// ```math
    /// \Delta x = \pm\sqrt{\frac{2u}{k}}
    /// ```
    fn extensions_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Length>; 2] {
        let extension =
            Quantity::<Length>::new((2.0 * energy.value() / self.stiffness.value()).sqrt());
        [extension, -extension]
    }
    /// ```math
    /// c(f) = \frac{1}{k}
    /// ```
    fn compliance(&self, _force: Quantity<Force>) -> Quantity<ReciprocalForcePerLength> {
        1.0 / self.stiffness
    }
    /// ```math
    /// \text{arg max }u(x) = \infty
    /// ```
    fn peak(&self) -> Quantity<Length> {
        Quantity::new(Scalar::INFINITY)
    }
    /// ```math
    /// f(x_\mathrm{peak}) = \infty
    /// ```
    fn peak_force(&self) -> Quantity<Force> {
        Quantity::new(Scalar::INFINITY)
    }
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Quantity<Length> {
        self.rest_length
    }
}
