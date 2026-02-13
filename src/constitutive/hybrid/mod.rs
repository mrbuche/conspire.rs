//! Hybrid constitutive models.

mod elastic;
mod elastic_viscoplastic;
mod hyperelastic;
mod hyperelastic_viscoplastic;

pub use elastic::{additive::ElasticAdditive, multiplicative::ElasticMultiplicative};

use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

// #[test]
// fn todo() {
//     todo!("Might as well get rid of the From impls once done since cannot get them through Deref.")
// }

/// A hybrid constitutive model based on the additive decomposition.
#[derive(Clone)]
pub struct Additive<C1, C2>(C1, C2);

impl<C1, C2> From<(C1, C2)> for Additive<C1, C2> {
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self(constitutive_model_1, constitutive_model_2)
    }
}

/// A hybrid constitutive model based on the multiplicative decomposition.
#[derive(Clone)]
pub struct Multiplicative<C1, C2>(C1, C2);

impl<C1, C2> From<(C1, C2)> for Multiplicative<C1, C2> {
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self(constitutive_model_1, constitutive_model_2)
    }
}

impl<C1, C2> Debug for Additive<C1, C2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Additive({}, {})",
            type_name::<C1>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            type_name::<C2>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap()
        )
    }
}

impl<C1, C2> Debug for Multiplicative<C1, C2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Multiplicative({}, {})",
            type_name::<C1>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            type_name::<C2>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap()
        )
    }
}
