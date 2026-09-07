//! The canonical construction of a constitutive model from two constituents.

use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

#[derive(Clone)]
pub struct Canonical<C1, C2>(pub(crate) C1, pub(crate) C2);

impl<C1, C2> From<(C1, C2)> for Canonical<C1, C2> {
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self(constitutive_model_1, constitutive_model_2)
    }
}

impl<C1, C2> Debug for Canonical<C1, C2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Canonical({}, {})", base_name::<C1>(), base_name::<C2>())
    }
}

fn base_name<T>() -> &'static str {
    type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap()
        .split('<')
        .next()
        .unwrap()
}
