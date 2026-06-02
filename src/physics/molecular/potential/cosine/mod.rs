use crate::{
    math::{Quantity, Scalar},
    physics::molecular::potential::AngularPotential,
    units::Energy,
};

/// The cosine potential.
#[derive(Clone, Debug)]
pub struct Cosine {
    /// The rest angle $`\theta_0`$.
    pub rest_angle: Scalar,
    /// The stiffness $`k`$.
    pub stiffness: Scalar,
}

impl AngularPotential for Cosine {
    /// ```math
    /// u(\theta) = k\left[1 - \cos(\theta - \theta_0)\right]
    /// ```
    fn energy(&self, angle: Scalar) -> Quantity<Energy> {
        Quantity::new(self.stiffness * (1.0 - (angle - self.rest_angle).cos()))
    }
}
