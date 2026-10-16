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
fn inertial_units_close_over_their_products() {
    use crate::units::{
        Acceleration, Density, Force, ForcePerLength, Mass, RateSquared, StressPerLength, Time,
        Velocity, Volume,
    };
    same::<Product<Density, Volume>, Mass>();
    same::<Product<Volume, Density>, Mass>();
    same::<Quotient<Mass, Volume>, Density>();
    same::<Quotient<Mass, Density>, Volume>();
    same::<Product<Mass, Acceleration>, Force>();
    same::<Quotient<Force, Mass>, Acceleration>();
    same::<Quotient<Force, Acceleration>, Mass>();
    same::<Product<Velocity, Rate>, Acceleration>();
    same::<Product<Length, RateSquared>, Acceleration>();
    same::<Quotient<Velocity, Time>, Acceleration>();
    same::<Product<Acceleration, Time>, Velocity>();
    same::<Product<Mass, RateSquared>, ForcePerLength>();
    same::<Product<Density, Acceleration>, StressPerLength>();
    same::<Product<Mass, Dimensionless>, Mass>();
    same::<Product<Dimensionless, Density>, Density>();
}

#[test]
fn specific_energy_closes_over_its_products() {
    use crate::units::{Density, Energy, Mass, SpecificEnergy, UnitRoot, Velocity};
    type Root<A> = <A as UnitRoot>::Output;
    same::<Product<Velocity, Velocity>, SpecificEnergy>();
    same::<Product<Mass, SpecificEnergy>, Energy>();
    same::<Product<SpecificEnergy, Mass>, Energy>();
    same::<Quotient<Energy, Mass>, SpecificEnergy>();
    same::<Quotient<Energy, SpecificEnergy>, Mass>();
    same::<Product<Density, SpecificEnergy>, Stress>();
    same::<Quotient<Stress, Density>, SpecificEnergy>();
    same::<Root<SpecificEnergy>, Velocity>();
}

#[test]
fn electrical_units_close_over_their_products() {
    use crate::units::{
        Area, Capacitance, CapacitancePerArea, CapacitancePerVolume, Charge, ChargeDensity,
        ElectricDisplacement, ElectricField, ElectricFieldSquared, ElectricPotential, Energy,
        Permittivity, Volume,
    };
    same::<Product<ElectricPotential, Charge>, Energy>();
    same::<Quotient<Energy, Charge>, ElectricPotential>();
    same::<Quotient<Energy, ElectricPotential>, Charge>();
    same::<Product<Capacitance, ElectricPotential>, Charge>();
    same::<Quotient<Charge, ElectricPotential>, Capacitance>();
    same::<Quotient<Charge, Capacitance>, ElectricPotential>();
    same::<Product<ElectricPotential, ReciprocalLength>, ElectricField>();
    same::<Quotient<ElectricPotential, Length>, ElectricField>();
    same::<Product<ElectricField, Length>, ElectricPotential>();
    same::<Product<Permittivity, ElectricField>, ElectricDisplacement>();
    same::<Quotient<ElectricDisplacement, ElectricField>, Permittivity>();
    same::<Quotient<ElectricDisplacement, Permittivity>, ElectricField>();
    same::<Product<Permittivity, Length>, Capacitance>();
    same::<Quotient<Capacitance, Length>, Permittivity>();
    same::<Product<Permittivity, ReciprocalLength>, CapacitancePerArea>();
    same::<Product<CapacitancePerArea, ReciprocalLength>, CapacitancePerVolume>();
    same::<Product<CapacitancePerArea, Area>, Capacitance>();
    same::<Product<CapacitancePerVolume, Volume>, Capacitance>();
    same::<Product<ElectricDisplacement, Area>, Charge>();
    same::<Quotient<Charge, Area>, ElectricDisplacement>();
    same::<Product<ElectricDisplacement, ReciprocalLength>, ChargeDensity>();
    same::<Product<ChargeDensity, Volume>, Charge>();
    same::<Quotient<Charge, Volume>, ChargeDensity>();
    same::<Product<ElectricField, ElectricDisplacement>, Stress>();
    same::<Product<ElectricField, ElectricField>, ElectricFieldSquared>();
    same::<Product<Permittivity, ElectricFieldSquared>, Stress>();
    same::<Quotient<Stress, Permittivity>, ElectricFieldSquared>();
    same::<Quotient<Stress, ElectricFieldSquared>, Permittivity>();
    same::<Quotient<Stress, ElectricField>, ElectricDisplacement>();
    same::<Product<Capacitance, Dimensionless>, Capacitance>();
    same::<Product<Dimensionless, ChargeDensity>, ChargeDensity>();
}

#[test]
fn conduction_units_close_over_their_products() {
    use crate::units::{
        Area, Charge, Conductance, ConductancePerArea, ConductancePerVolume, Current,
        CurrentDensity, CurrentPerVolume, ElectricField, ElectricFieldSquared, ElectricPotential,
        ElectricalConductivity, Power, PowerDensity, Time, Volume,
    };
    same::<Product<Current, Time>, Charge>();
    same::<Quotient<Charge, Time>, Current>();
    same::<Quotient<Charge, Current>, Time>();
    same::<Product<Current, ElectricPotential>, Power>();
    same::<Quotient<Power, ElectricPotential>, Current>();
    same::<Quotient<Power, Current>, ElectricPotential>();
    same::<Product<Conductance, ElectricPotential>, Current>();
    same::<Quotient<Current, ElectricPotential>, Conductance>();
    same::<Quotient<Current, Conductance>, ElectricPotential>();
    same::<Product<ElectricalConductivity, ElectricField>, CurrentDensity>();
    same::<Quotient<CurrentDensity, ElectricField>, ElectricalConductivity>();
    same::<Quotient<CurrentDensity, ElectricalConductivity>, ElectricField>();
    same::<Product<ElectricalConductivity, Length>, Conductance>();
    same::<Quotient<Conductance, Length>, ElectricalConductivity>();
    same::<Product<ElectricalConductivity, ReciprocalLength>, ConductancePerArea>();
    same::<Product<ConductancePerArea, ReciprocalLength>, ConductancePerVolume>();
    same::<Product<ConductancePerArea, Area>, Conductance>();
    same::<Product<ConductancePerVolume, Volume>, Conductance>();
    same::<Product<CurrentDensity, Area>, Current>();
    same::<Quotient<Current, Area>, CurrentDensity>();
    same::<Product<CurrentDensity, ReciprocalLength>, CurrentPerVolume>();
    same::<Product<CurrentPerVolume, Volume>, Current>();
    same::<Quotient<Current, Volume>, CurrentPerVolume>();
    same::<Product<ElectricalConductivity, ElectricFieldSquared>, PowerDensity>();
    same::<Quotient<PowerDensity, ElectricalConductivity>, ElectricFieldSquared>();
    same::<Product<ElectricField, CurrentDensity>, PowerDensity>();
    same::<Quotient<PowerDensity, ElectricField>, CurrentDensity>();
    same::<Product<Conductance, Dimensionless>, Conductance>();
    same::<Product<Dimensionless, CurrentPerVolume>, CurrentPerVolume>();
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
