/// The configuration a tensor index belongs to.
///
/// Indices may only be contracted when their configurations agree, which is
/// checked when the tensor operations are compiled rather than when they run.
pub trait Configuration {}

macro_rules! configurations {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct $name;
            impl Configuration for $name {}
        )+
    };
}

configurations!(
    /// The reference configuration.
    Reference,
    /// The current configuration.
    Current,
    /// An intermediate configuration.
    Intermediate,
    /// A second intermediate configuration.
    Auxiliary,
    /// The space a composite element projects onto.
    Projection,
    /// The shared index of a factorization.
    Factor,
    /// The second index of a flattened higher-rank tensor.
    Flattened,
);
