//! The [`Autodiff`] wrapper, shared across constitutive families.

/// Wraps a model that supplies `#[autodiff]`-differentiable scalar kernels,
/// giving it the hand-written constitutive trait APIs via Enzyme.
#[derive(Clone, Debug)]
pub struct Autodiff<M>(pub M);
