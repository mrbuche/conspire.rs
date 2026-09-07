//! The [`Autodiff`] wrapper, shared across constitutive families.

/// Wraps a model that supplies `#[autodiff]`-differentiable scalar kernels,
/// giving it the hand-written constitutive trait APIs via Enzyme. The trait
/// impls live with the traits they satisfy — see
/// [`solid::elastic::autodiff`](crate::constitutive::solid::elastic::autodiff)
/// and
/// [`solid::hyperelastic::autodiff`](crate::constitutive::solid::hyperelastic::autodiff).
#[derive(Clone, Debug)]
pub struct Autodiff<M>(pub M);
