use crate::{math::Scalar, physics::molecular::potential::Potential};

#[derive(Clone, Debug)]
pub struct Harmonic {
    pub rest_length: Scalar,
    pub stiffness: Scalar,
}

impl Potential for Harmonic {
    fn energy(&self, length: Scalar) -> Scalar {
        0.5 * self.stiffness * (length - self.rest_length).powi(2)
    }
    fn force(&self, length: Scalar) -> Scalar {
        self.stiffness * (length - self.rest_length)
    }
    fn stiffness(&self, _length: Scalar) -> Scalar {
        self.stiffness
    }
    fn extension(&self, force: Scalar) -> Scalar {
        force / self.stiffness
    }
    fn compliance(&self, force: Scalar) -> Scalar {
        1.0 / self.stiffness
    }
    fn maximum_force(&self) -> Scalar {
        Scalar::INFINITY
    }
    fn peak(&self) -> Scalar {
        Scalar::INFINITY
    }
    fn rest_length(&self) -> Scalar {
        self.rest_length
    }
}
