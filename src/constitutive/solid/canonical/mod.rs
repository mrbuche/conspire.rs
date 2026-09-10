use crate::{
    constitutive::{canonical::Canonical, solid::Solid},
    math::Quantity,
    units::Stress,
};

impl<C1, C2> Solid for Canonical<C1, C2>
where
    C1: Solid,
    C2: Clone,
{
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.0.bulk_modulus()
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.0.shear_modulus()
    }
}
