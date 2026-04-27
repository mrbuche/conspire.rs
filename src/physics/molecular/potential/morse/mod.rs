use crate::{math::Scalar, physics::molecular::potential::Potential};

/// The Morse potential.[^1]
/// [^1]: P.M. Morse, [Physical Review **34**, 57 (1929)](https://doi.org/10.1103/PhysRev.34.57).
#[derive(Clone, Debug)]
pub struct Morse {
    /// The rest length $`x_0`$.
    pub rest_length: Scalar,
    /// The potential depth $`u_0`$.
    pub depth: Scalar,
    /// The Morse parameter $`a`$.
    pub parameter: Scalar,
}

impl Potential for Morse {
    /// ```math
    /// u(x) = u_0\left[1 - e^{-a(x - x_0)}\right]^2
    /// ```
    fn energy(&self, length: Scalar) -> Scalar {
        self.depth * (1.0 - (self.parameter * (self.rest_length - length)).exp()).powi(2)
    }
    /// ```math
    /// f(x) = 2au_0e^{-a(x - x_0)}\left[1 - e^{-a(x - x_0)}\right]
    /// ```
    fn force(&self, length: Scalar) -> Scalar {
        let exp = (self.parameter * (self.rest_length - length)).exp();
        2.0 * self.parameter * self.depth * exp * (1.0 - exp)
    }
    /// ```math
    /// f(u) = \pm 2a\sqrt{u/u_0}\left(1 \mp \sqrt{u/u_0}\right)
    /// ```
    fn force_at_energy(&self, energy: Scalar) -> Scalar {
        let y = energy / self.depth;
        if (0.0..=1.0).contains(&y) {
            2.0 * self.parameter * y.sqrt() * (1.0 - y.sqrt())
        } else {
            Scalar::NAN
        }
    }
    /// ```math
    /// k(x) = 2a^2u_0e^{-a(x - x_0)}\left[2e^{-a(x - x_0)} - 1\right]
    /// ```
    fn stiffness(&self, length: Scalar) -> Scalar {
        let exp = (self.parameter * (self.rest_length - length)).exp();
        2.0 * self.parameter.powi(2) * self.depth * exp * (2.0 * exp - 1.0)
    }
    /// ```math
    /// h(x) = 2a^3u_0e^{-a(x - x_0)}\left[1 - 4e^{-a(x - x_0)}\right]
    /// ```
    fn anharmonicity(&self, length: Scalar) -> Scalar {
        let exp = (self.parameter * (self.rest_length - length)).exp();
        2.0 * self.parameter.powi(3) * self.depth * exp * (1.0 - 4.0 * exp)
    }
    /// ```math
    /// \Delta x(f) = \frac{1}{a}\,\ln\left(\frac{2}{1 + \sqrt{1 - f/f_\mathrm{max}}}\right)
    /// ```
    fn extension(&self, force: Scalar) -> Scalar {
        let y = force / self.peak_force();
        if (0.0..=1.0).contains(&y) {
            (2.0 / (1.0 + (1.0 - y).sqrt())).ln() / self.parameter
        } else {
            Scalar::NAN
        }
    }
    /// ```math
    /// \Delta x(u) = \frac{1}{a}\,\ln\left(\frac{1}{1\mp\sqrt{u/u_0}}\right)
    /// ```
    fn extension_at_energy(&self, energy: Scalar) -> Scalar {
        let y = energy / self.depth;
        if (0.0..=1.0).contains(&y) {
            (1.0 / (1.0 - y.sqrt())).ln() / self.parameter
        } else {
            Scalar::NAN
        }
    }
    /// ```math
    /// c(f) = \frac{1}{a^2u_0}\,\frac{\left(1-f/f_\mathrm{max}\right)^{-1/2}}{1+\sqrt{1-f/f_\mathrm{max}}}
    /// ```
    fn compliance(&self, force: Scalar) -> Scalar {
        let y = force / self.peak_force();
        if (0.0..1.0).contains(&y) {
            let s = (1.0 - y).sqrt();
            1.0 / (self.parameter.powi(2) * self.depth) / (s * (1.0 + s))
        } else if y == 0.0 {
            Scalar::INFINITY
        } else {
            Scalar::NAN
        }
    }
    /// ```math
    /// \text{arg max }u(x) = x_0 + \frac{1}{a}\,\ln(2)
    /// ```
    fn peak(&self) -> Scalar {
        self.rest_length + 2.0_f64.ln() / self.parameter
    }
    /// ```math
    /// f(x_\mathrm{peak}) = \frac{au_0}{2}
    /// ```
    fn peak_force(&self) -> Scalar {
        0.5 * self.parameter * self.depth
    }
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Scalar {
        self.rest_length
    }
}
