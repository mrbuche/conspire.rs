use crate::{
    math::{
        Quantity, Scalar,
        unit::{
            Energy, Force, ForcePerLength, Length, ReciprocalForcePerLength, ReciprocalLength,
            Stress,
        },
    },
    physics::molecular::potential::Potential,
};

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

impl Morse {
    /// Returns the potential depth.
    fn depth(&self) -> Quantity<Energy> {
        self.depth.into()
    }
    /// Returns the Morse parameter.
    fn parameter(&self) -> Quantity<ReciprocalLength> {
        self.parameter.into()
    }
}

impl Potential for Morse {
    /// ```math
    /// u(x) = u_0\left[1 - e^{-a(x - x_0)}\right]^2
    /// ```
    fn energy(&self, length: Quantity<Length>) -> Quantity<Energy> {
        let exp = (self.parameter() * (self.rest_length() - length)).exp();
        self.depth() * (1.0 - exp).powi(2)
    }
    /// ```math
    /// f(x) = 2au_0e^{-a(x - x_0)}\left[1 - e^{-a(x - x_0)}\right]
    /// ```
    fn force(&self, length: Quantity<Length>) -> Quantity<Force> {
        let exp = (self.parameter() * (self.rest_length() - length)).exp();
        2.0 * self.parameter() * self.depth() * exp * (1.0 - exp)
    }
    /// ```math
    /// f(u) = \pm 2a u_0\sqrt{u/u_0}\left(1 \mp \sqrt{u/u_0}\right)
    /// ```
    fn forces_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Force>; 2] {
        let y = energy / self.depth();
        let f = 2.0 * self.parameter() * self.depth() * y.sqrt();
        let tensile = if (0.0..=1.0).contains(&y) {
            f * (1.0 - y.sqrt())
        } else {
            Quantity::new(Scalar::NAN)
        };
        [tensile, -f * (1.0 + y.sqrt())]
    }
    /// ```math
    /// k(x) = 2a^2u_0e^{-a(x - x_0)}\left[2e^{-a(x - x_0)} - 1\right]
    /// ```
    fn stiffness(&self, length: Quantity<Length>) -> Quantity<ForcePerLength> {
        let exp = (self.parameter() * (self.rest_length() - length)).exp();
        2.0 * (self.parameter() * self.parameter()) * self.depth() * exp * (2.0 * exp - 1.0)
    }
    /// ```math
    /// h(x) = 2a^3u_0e^{-a(x - x_0)}\left[1 - 4e^{-a(x - x_0)}\right]
    /// ```
    fn anharmonicity(&self, length: Quantity<Length>) -> Quantity<Stress> {
        let exp = (self.parameter() * (self.rest_length() - length)).exp();
        2.0 * (self.parameter() * self.parameter())
            * self.depth()
            * self.parameter()
            * exp
            * (1.0 - 4.0 * exp)
    }
    /// ```math
    /// \Delta x(f) = \frac{1}{a}\,\ln\left(\frac{2}{1 + \sqrt{1 - f/f_\mathrm{max}}}\right)
    /// ```
    fn extension(&self, force: Quantity<Force>) -> Quantity<Length> {
        let y = force / self.peak_force();
        if y <= 1.0 {
            (2.0 / (1.0 + (1.0 - y).sqrt())).ln() / self.parameter()
        } else {
            Quantity::new(Scalar::NAN)
        }
    }
    /// ```math
    /// \Delta x(u) = \frac{1}{a}\,\ln\left(\frac{1}{1\mp\sqrt{u/u_0}}\right)
    /// ```
    fn extensions_at_energy(&self, energy: Quantity<Energy>) -> [Quantity<Length>; 2] {
        let y = energy / self.depth();
        let tensile = if (0.0..=1.0).contains(&y) {
            (1.0 / (1.0 - y.sqrt())).ln() / self.parameter()
        } else {
            Quantity::new(Scalar::NAN)
        };
        [tensile, (1.0 / (1.0 + y.sqrt())).ln() / self.parameter()]
    }
    /// ```math
    /// c(f) = \frac{1}{a^2u_0}\,\frac{\left(1-f/f_\mathrm{max}\right)^{-1/2}}{1+\sqrt{1-f/f_\mathrm{max}}}
    /// ```
    fn compliance(&self, force: Quantity<Force>) -> Quantity<ReciprocalForcePerLength> {
        let y = force / self.peak_force();
        if (0.0..1.0).contains(&y) {
            let s = (1.0 - y).sqrt();
            1.0 / (self.parameter() * self.parameter() * self.depth()) / (s * (1.0 + s))
        } else if y == 0.0 {
            Quantity::new(Scalar::INFINITY)
        } else {
            Quantity::new(Scalar::NAN)
        }
    }
    /// ```math
    /// \text{arg max }u(x) = x_0 + \frac{1}{a}\,\ln(2)
    /// ```
    fn peak(&self) -> Quantity<Length> {
        self.rest_length() + 2.0_f64.ln() / self.parameter()
    }
    /// ```math
    /// f(x_\mathrm{peak}) = \frac{au_0}{2}
    /// ```
    fn peak_force(&self) -> Quantity<Force> {
        0.5 * self.parameter() * self.depth()
    }
    /// ```math
    /// \text{arg min }u(x) = x_0
    /// ```
    fn rest_length(&self) -> Quantity<Length> {
        self.rest_length.into()
    }
}
