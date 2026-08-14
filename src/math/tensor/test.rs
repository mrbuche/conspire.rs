use crate::math::{Current, Reference, Tensor, TensorArray, TensorRank2};
use crate::units::{Dimensionless, Rate, Stress, Viscosity};

type Deformation = TensorRank2<3, Current, Reference, Dimensionless>;
type Rates = TensorRank2<3, Reference, Reference, Rate>;
type Viscosities = TensorRank2<3, Current, Reference, Viscosity>;
type Stresses = TensorRank2<3, Current, Reference, Stress>;

#[test]
fn a_unit_costs_no_space() {
    assert_eq!(size_of::<Stresses>(), size_of::<Deformation>());
}

#[test]
fn same_units_add() {
    let sum = Stresses::zero() + Stresses::zero();
    assert!(sum.is_zero())
}

#[test]
fn multiplication_combines_the_units() {
    // Viscosity * Rate = Stress, resolved when this compiles.
    let stress: Stresses = Viscosities::zero() * Rates::zero();
    assert!(stress.is_zero())
}

#[test]
fn the_default_is_dimensionless() {
    let product =
        TensorRank2::<3, Current, Reference>::zero() * TensorRank2::<3, Reference, Current>::zero();
    assert!(product.is_zero())
}
