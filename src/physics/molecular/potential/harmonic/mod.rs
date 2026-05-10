use crate::{
    math::{Scalar, ScalarList},
    physics::molecular::potential::Potential,
};

/// The harmonic potential.
#[derive(Clone, Debug)]
pub struct Harmonic {
    /// The rest length $`x_0`$.
    pub rest_length: Scalar,
    /// The stiffness $`k`$.
    pub stiffness: Scalar,
}

impl Potential for Harmonic {
    /// ```math
    /// u(x) = \frac{1}{2}\,k(x - x_0)^2
    /// ```
    fn energy(&self, length: Scalar) -> Scalar {
        0.5 * self.stiffness * (length - self.rest_length).powi(2)
    }
    /// ```math
    /// f(x) = k(x - x_0)
    /// ```
    fn force(&self, length: Scalar) -> Scalar {
        self.stiffness * (length - self.rest_length)
    }
    /// ```math
    /// f = \pm\sqrt{2ku}
    /// ```
    fn forces_at_energy(&self, energy: Scalar) -> ScalarList<2> {
        let force = (2.0 * self.stiffness * energy).sqrt();
        [force, -force].into()
    }
    /// ```math
    /// k(x) = k
    /// ```
    fn stiffness(&self, _length: Scalar) -> Scalar {
        self.stiffness
    }
    /// ```math
    /// h(x) = 0.0
    /// ```
    fn anharmonicity(&self, _length: Scalar) -> Scalar {
        0.0
    }
    /// ```math
    /// \Delta x(f) = \frac{f}{k}
    /// ```
    fn extension(&self, force: Scalar) -> Scalar {
        force / self.stiffness
    }
    /// ```math
    /// \Delta x = \pm\sqrt{\frac{2u}{k}}
    /// ```
    fn extensions_at_energy(&self, energy: Scalar) -> ScalarList<2> {
        let extension = (2.0 * energy / self.stiffness).sqrt();
        [extension, -extension].into()
    }
    /// ```math
    /// c(f) = \frac{1}{k}
    /// ```
    fn compliance(&self, _force: Scalar) -> Scalar {
        1.0 / self.stiffness
    }
    /// ```math
    /// \text{arg max }u(x) = \infty
    /// ```
    fn peak(&self) -> Scalar {
        Scalar::INFINITY
    }
    /// ```math
    /// f(x_\mathrm{peak}) = \infty
    /// ```
    fn peak_force(&self) -> Scalar {
        Scalar::INFINITY
    }
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Scalar {
        self.rest_length
    }
}
