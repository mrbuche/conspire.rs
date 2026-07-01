use super::{Scalar, Tensor};

/// Different norms for tensors.
#[derive(Debug, Default)]
pub enum Norm {
    Chebyshev,
    #[default]
    Euclidean,
    Manhattan,
    Minkowski(Scalar),
}

impl Norm {
    pub fn apply<T: Tensor>(&self, t: &T) -> Scalar {
        match self {
            Self::Chebyshev => t.norm_inf(),
            Self::Euclidean => t.norm(),
            Self::Manhattan => t.norm_l1(),
            Self::Minkowski(p) => t.norm_p(*p),
        }
    }
}
