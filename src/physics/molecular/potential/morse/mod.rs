use crate::{math::Scalar, physics::molecular::potential::Potential};

#[derive(Clone, Debug)]
pub struct Morse {
    pub rest_length: Scalar,
    pub depth: Scalar,
    pub parameter: Scalar,
}

impl Potential for Morse {
    fn energy(&self, length: Scalar) -> Scalar {
        self.depth * (1.0 - (self.parameter * (self.rest_length - length)).exp()).powi(2)
    }
    fn force(&self, length: Scalar) -> Scalar {
        let exp = (self.parameter * (self.rest_length - length)).exp();
        2.0 * self.parameter * self.depth * exp * (1.0 - exp)
    }
    fn stiffness(&self, length: Scalar) -> Scalar {
        let exp = (self.parameter * (self.rest_length - length)).exp();
        2.0 * self.parameter.powi(2) * self.depth * exp * (2.0 * exp - 1.0)
    }
    /// ```math
    /// \Delta x = \frac{1}{a}\,\ln\left(\frac{2}{1 \pm \sqrt{1 - f/f_\mathrm{max}}}\right)
    /// ```
    fn extension(&self, force: Scalar) -> Scalar {
        let y = force / self.maximum_force();
        if y < 1.0 {
            (2.0 / (1.0 + (1.0 - y).sqrt()))
                .ln() / self.parameter
        } else {
            (2.0 / (1.0 - (1.0 - y).sqrt()))
                .ln() / self.parameter
        }
    }
    fn compliance(&self, force: Scalar) -> Scalar {
        todo!()
    }
    /// ```math
    /// f_\mathrm{max} = \frac{au_0}{2}
    /// ```
    fn maximum_force(&self) -> Scalar {
        0.5 * self.parameter * self.depth
    }
    fn peak(&self) -> Scalar {
        self.rest_length + 2.0_f64.ln() / self.parameter
    }
    fn rest_length(&self) -> Scalar {
        self.rest_length
    }
}
