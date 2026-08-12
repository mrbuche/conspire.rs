#[cfg(test)]
mod test;

use super::{Quantity, Scalar, Tensor};

/// Different norms for tensors.
#[derive(Clone, Copy, Debug, Default)]
pub enum Norm {
    Chebyshev,
    #[default]
    Euclidean,
    Manhattan,
    Minkowski(Scalar),
}

impl Norm {
    /// The norm of a whole tensor, which carries the unit the tensor does.
    ///
    /// Every variant is that unit once: the Minkowski norm takes the p-th root
    /// of a sum of p-th powers, so only the sum inside it names no unit.
    pub fn apply<T: Tensor>(&self, t: &T) -> Quantity<T::Unit> {
        match self {
            Self::Chebyshev => t.norm_inf(),
            Self::Euclidean => t.norm(),
            Self::Manhattan => t.norm_l1(),
            Self::Minkowski(p) => t.norm_p(*p),
        }
    }
    /// The norm of a whole tensor as a number, for comparing against a
    /// tolerance the caller supplies in whatever units the problem is in.
    pub fn measure<T: Tensor>(&self, t: &T) -> Scalar {
        self.apply(t).value()
    }
    /// The norm of some values, for when they are part of a tensor rather than
    /// all of one.
    pub fn over(&self, values: impl Iterator<Item = Scalar>) -> Scalar {
        match self {
            Self::Chebyshev => values.fold(0.0, |largest: Scalar, value| largest.max(value.abs())),
            Self::Euclidean => values.map(|value| value * value).sum::<Scalar>().sqrt(),
            Self::Manhattan => values.map(|value| value.abs()).sum(),
            Self::Minkowski(p) => values
                .map(|value| value.abs().powf(*p))
                .sum::<Scalar>()
                .powf(1.0 / p),
        }
    }
}
