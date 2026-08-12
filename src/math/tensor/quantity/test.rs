use super::Quantity;
use crate::math::unit::{Dimensionless, Rate, Stress, Viscosity};

type Modulus = Quantity<Stress>;
type Viscous = Quantity<Viscosity>;

#[test]
fn scaling_by_a_bare_scalar_keeps_the_unit() {
    let doubled = 2.0 * Modulus::new(3.0);
    assert_eq!(doubled, Modulus::new(6.0));
    assert_eq!(doubled / 3.0, Modulus::new(2.0))
}

#[test]
fn same_units_add() {
    assert_eq!(Modulus::new(1.0) + Modulus::new(2.0), Modulus::new(3.0))
}

#[test]
fn dividing_a_stress_by_a_viscosity_is_a_rate() {
    let rate: Quantity<Rate> = Modulus::new(6.0) / Viscous::new(2.0);
    assert_eq!(rate, Quantity::<Rate>::new(3.0))
}

#[test]
fn a_unit_costs_no_space() {
    assert_eq!(size_of::<Modulus>(), size_of::<f64>());
    assert_eq!(size_of::<Quantity<Dimensionless>>(), size_of::<f64>())
}

mod constitutive_law {
    use crate::math::unit::{Dimensionless, Rate, Stress, Viscosity};
    use crate::math::{Current, Quantity, Tensor, TensorArray, TensorRank2};

    type Strain = TensorRank2<3, Current, Current, Dimensionless>;
    type StrainRate = TensorRank2<3, Current, Current, Rate>;
    type Stresses = TensorRank2<3, Current, Current, Stress>;

    /// The shape of `almansi_hamel`'s Cauchy stress, which could not be typed
    /// before the material parameters carried their units: an elastic term and
    /// a viscous term that are only the same quantity because a modulus is a
    /// stress and a viscosity is a stress per unit rate.
    #[test]
    fn an_elastic_and_a_viscous_term_are_both_stresses() {
        let shear_modulus = Quantity::<Stress>::new(2.0);
        let shear_viscosity = Quantity::<Viscosity>::new(3.0);
        let jacobian = 1.5;
        let stress: Stresses = Strain::identity() * (2.0 * shear_modulus / jacobian)
            + StrainRate::identity() * (2.0 * shear_viscosity / jacobian);
        assert_eq!(stress[0][0], 2.0 * 2.0 / 1.5 + 2.0 * 3.0 / 1.5)
    }

    #[test]
    fn dividing_a_stress_by_a_viscosity_gives_a_rate() {
        let rate = Stresses::identity() / Quantity::<Viscosity>::new(2.0);
        assert_eq!(
            rate.norm_squared(),
            StrainRate::identity().norm_squared() / 4.0
        )
    }
}
