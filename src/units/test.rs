use crate::units::{
    Dimensionless, Length, Rate, ReciprocalLength, Stress, UnitDiv, UnitInv, UnitMul, Viscosity,
};

trait Same<T> {}
impl<T> Same<T> for T {}

/// Compiles only when the two types are the same.
fn same<A: Same<B>, B>() {}

type Product<A, B> = <A as UnitMul<B>>::Output;
type Quotient<A, B> = <A as UnitDiv<B>>::Output;
type Inverse<A> = <A as UnitInv>::Output;

#[test]
fn dimensionless_is_the_identity() {
    same::<Product<Stress, Dimensionless>, Stress>();
    same::<Product<Dimensionless, Stress>, Stress>();
    same::<Product<Dimensionless, Dimensionless>, Dimensionless>();
}

#[test]
fn viscosity_times_rate_is_stress() {
    same::<Product<Viscosity, Rate>, Stress>();
    same::<Product<Rate, Viscosity>, Stress>();
}

#[test]
fn division_undoes_multiplication() {
    same::<Quotient<Stress, Rate>, Viscosity>();
    same::<Quotient<Product<Viscosity, Rate>, Rate>, Viscosity>();
}

#[test]
fn length_and_its_reciprocal_cancel() {
    same::<Product<Length, ReciprocalLength>, Dimensionless>();
    same::<Inverse<Length>, ReciprocalLength>();
    same::<Inverse<Inverse<Length>>, Length>();
}

#[test]
fn units_are_zero_sized() {
    assert_eq!(size_of::<Stress>(), 0);
    assert_eq!(size_of::<Dimensionless>(), 0);
}

mod synonyms {
    use crate::units::{
        Compliance, EnergyDensity, Fluidity, Frequency, Modulus, Pressure, Rate, ReciprocalStress,
        ReciprocalTemperature, ReciprocalViscosity, Stress, ThermalExpansion,
    };

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn same<A: Same<B>, B>() {}

    /// A dimensional system cannot tell these apart, and should not try.
    #[test]
    fn a_synonym_denotes_the_same_unit() {
        same::<Pressure, Stress>();
        same::<EnergyDensity, Stress>();
        same::<Modulus, Stress>();
        same::<Compliance, ReciprocalStress>();
        same::<Fluidity, ReciprocalViscosity>();
        same::<ThermalExpansion, ReciprocalTemperature>();
        same::<Frequency, Rate>();
    }
}
