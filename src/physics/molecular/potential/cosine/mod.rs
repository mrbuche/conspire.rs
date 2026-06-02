use crate::{
    math::{Scalar, ScalarList},
    physics::molecular::potential::Potential,
};
use std::f64::consts::PI;

/// The cosine potential.
#[derive(Clone, Debug)]
pub struct Cosine {
    /// The rest angle $`x_0`$.
    pub rest_length: Scalar,
    /// The stiffness $`k`$.
    pub stiffness: Scalar,
}

impl Potential for Cosine {
    /// ```math
    /// u(x) = k\left[1 - \cos(x - x_0)\right]
    /// ```
    fn energy(&self, length: Scalar) -> Scalar {
        self.stiffness * (1.0 - (length - self.rest_length).cos())
    }
    /// ```math
    /// f(x) = k\sin(x - x_0)
    /// ```
    fn force(&self, length: Scalar) -> Scalar {
        self.stiffness * (length - self.rest_length).sin()
    }
    /// ```math
    /// f = \pm\sqrt{u(2k - u)}
    /// ```
    fn forces_at_energy(&self, energy: Scalar) -> ScalarList<2> {
        if (0.0..=2.0 * self.stiffness).contains(&energy) {
            let force = (energy * (2.0 * self.stiffness - energy)).sqrt();
            [force, -force].into()
        } else {
            [Scalar::NAN, Scalar::NAN].into()
        }
    }
    /// ```math
    /// k(x) = k\cos(x - x_0)
    /// ```
    fn stiffness(&self, length: Scalar) -> Scalar {
        self.stiffness * (length - self.rest_length).cos()
    }
    /// ```math
    /// h(x) = -k\sin(x - x_0)
    /// ```
    fn anharmonicity(&self, length: Scalar) -> Scalar {
        -self.stiffness * (length - self.rest_length).sin()
    }
    /// ```math
    /// \Delta x(f) = \arcsin(f/k)
    /// ```
    fn extension(&self, force: Scalar) -> Scalar {
        (force / self.stiffness).asin()
    }
    /// ```math
    /// \Delta x = \pm\arccos(1 - u/k)
    /// ```
    fn extensions_at_energy(&self, energy: Scalar) -> ScalarList<2> {
        let extension = (1.0 - energy / self.stiffness).acos();
        [extension, -extension].into()
    }
    /// ```math
    /// c(f) = \frac{1}{\sqrt{k^2 - f^2}}
    /// ```
    fn compliance(&self, force: Scalar) -> Scalar {
        1.0 / (self.stiffness.powi(2) - force.powi(2)).sqrt()
    }
    /// ```math
    /// \text{arg max }f(x) = x_0 + \frac{\pi}{2}
    /// ```
    fn peak(&self) -> Scalar {
        self.rest_length + 0.5 * PI
    }
    /// ```math
    /// f(x_\mathrm{peak}) = k
    /// ```
    fn peak_force(&self) -> Scalar {
        self.stiffness
    }
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Scalar {
        self.rest_length
    }
}
