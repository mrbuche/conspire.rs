use super::super::{
    Area, Dimensionless, Energy, Entropy, Force, ForcePerLength, Length, Rate, Stress,
    StressPerLength, Temperature, Time, Viscosity, Volume,
};
use crate::math::{Quantity, TensorRank0};

const EPSILON: TensorRank0 = 1e-12;

fn same<U>(one: Quantity<U>, two: Quantity<U>) {
    assert!(
        !one.differs(two, EPSILON),
        "{} against {}",
        one.value(),
        two.value()
    )
}

//
// A scale is spent where the quantity is built, so a constructor has to be
// usable where a constant is, which is the whole reason it is a `const fn`.
//
const MODULUS: Quantity<Stress> = Stress::gigapascals(1.0);

#[test]
fn a_scale_is_spent_at_compile_time() {
    same(MODULUS, Stress::pascals(1e9))
}

mod agree {
    use super::*;
    #[test]
    fn lengths() {
        same(Length::millimeters(1e3), Length::meters(1.0));
        same(Length::micrometers(1e3), Length::millimeters(1.0));
        same(Length::nanometers(1e3), Length::micrometers(1.0));
        same(Length::centimeters(1e2), Length::meters(1.0));
        same(Length::kilometers(1.0), Length::meters(1e3));
        same(Length::inches(1.0), Length::millimeters(25.4));
        same(Length::feet(1.0), Length::inches(12.0))
    }
    #[test]
    fn stresses() {
        same(Stress::kilopascals(1e3), Stress::megapascals(1.0));
        same(Stress::megapascals(1e3), Stress::gigapascals(1.0));
        same(Stress::bars(1.0), Stress::kilopascals(1e2));
        same(Stress::ksi(1.0), Stress::psi(1e3))
    }
    #[test]
    fn times() {
        same(Time::milliseconds(1e3), Time::seconds(1.0));
        same(Time::microseconds(1e3), Time::milliseconds(1.0));
        same(Time::minutes(1.0), Time::seconds(6e1));
        same(Time::hours(1.0), Time::minutes(6e1))
    }
    #[test]
    fn forces() {
        same(Force::millinewtons(1e3), Force::newtons(1.0));
        same(Force::kilonewtons(1e3), Force::meganewtons(1.0))
    }
    #[test]
    fn viscosities() {
        same(Viscosity::poise(1e1), Viscosity::pascal_seconds(1.0));
        same(Viscosity::centipoise(1e2), Viscosity::poise(1.0))
    }
}

//
// A factor is only right against the rest of the set, so the products that
// cross from one unit to another are what actually tests the table. A wrong
// factor shows up here and nowhere else.
//
mod cohere {
    use super::*;
    #[test]
    fn a_stress_over_an_area_is_a_force() {
        same(
            Stress::megapascals(1.0) * Area::square_millimeters(1.0),
            Force::newtons(1.0),
        );
        same(
            Stress::pascals(1.0) * Area::square_meters(1.0),
            Force::newtons(1.0),
        )
    }
    #[test]
    fn a_stress_over_a_time_is_a_viscosity() {
        same(
            Stress::pascals(1.0) * Time::seconds(1.0),
            Viscosity::pascal_seconds(1.0),
        )
    }
    #[test]
    fn a_rate_is_a_reciprocal_time() {
        same(
            Rate::per_second(1.0) * Time::seconds(1.0),
            Dimensionless::of(1.0),
        );
        same(
            Rate::per_hour(1.0) * Time::hours(1.0),
            Dimensionless::of(1.0),
        );
        same(Rate::hertz(1.0), Rate::per_second(1.0))
    }
    #[test]
    fn a_length_cubed_is_a_volume() {
        same(
            Length::meters(1.0) * Length::meters(1.0) * Length::meters(1.0),
            Volume::cubic_meters(1.0),
        );
        same(
            Length::millimeters(1.0) * Length::millimeters(1.0) * Length::millimeters(1.0),
            Volume::cubic_millimeters(1.0),
        );
        same(Volume::liters(1e3), Volume::cubic_meters(1.0))
    }
    #[test]
    fn a_length_squared_is_an_area() {
        same(
            Length::meters(1.0) * Length::meters(1.0),
            Area::square_meters(1.0),
        );
        same(
            Length::millimeters(1.0) * Length::millimeters(1.0),
            Area::square_millimeters(1.0),
        )
    }
    #[test]
    fn an_entropy_over_a_temperature_is_an_energy() {
        same(
            Entropy::joules_per_kelvin(1.0) * Temperature::kelvin(1.0),
            Energy::joules(1.0),
        )
    }
    #[test]
    fn a_force_over_a_length_is_an_energy() {
        same(
            Force::newtons(1.0) * Length::meters(1.0),
            Energy::joules(1.0),
        );
        same(
            Energy::electronvolts(1.0),
            Energy::joules(1.602_176_634e-19),
        )
    }
    #[test]
    fn a_force_per_length_is_a_force_over_a_length() {
        same(
            ForcePerLength::newtons_per_meter(1.0) * Length::meters(1.0),
            Force::newtons(1.0),
        );
        same(
            ForcePerLength::piconewtons_per_nanometer(1.0) * Length::meters(1.0),
            Force::millinewtons(1.0),
        )
    }
    #[test]
    fn a_stress_over_a_length_is_a_stress_per_length() {
        same(
            Stress::pascals(1.0) / Length::meters(1.0),
            StressPerLength::pascals_per_meter(1.0),
        );
        same(
            Stress::megapascals(1.0) / Length::millimeters(1.0),
            StressPerLength::megapascals_per_millimeter(1.0),
        )
    }
}

//
// A celsius is offset rather than scaled, which is the one thing a factor
// cannot say, and the reason the scale is spent here rather than carried.
//
mod offset {
    use super::*;
    #[test]
    fn celsius_is_kelvin_moved() {
        same(Temperature::celsius(0.0), Temperature::kelvin(273.15));
        same(Temperature::celsius(-273.15), Temperature::kelvin(0.0))
    }
    #[test]
    fn fahrenheit_is_celsius_moved_and_scaled() {
        same(Temperature::fahrenheit(32.0), Temperature::celsius(0.0));
        same(Temperature::fahrenheit(212.0), Temperature::celsius(1e2));
        same(Temperature::fahrenheit(-40.0), Temperature::celsius(-40.0))
    }
}

//
// A quantity is read back by naming a scale, not by supplying one, so the
// scale a value was written in never has to be kept alongside it.
//
mod read {
    use super::*;
    #[test]
    fn a_scale_read_back_is_the_number_written() {
        assert_eq!(Stress::megapascals(3.0).in_megapascals(), 3.0);
        assert_eq!(Length::millimeters(25.4).in_millimeters(), 25.4);
        assert_eq!(Time::hours(1.5).in_hours(), 1.5)
    }
    #[test]
    fn a_scale_reads_back_in_any_other() {
        assert_eq!(Stress::megapascals(3.0).in_pascals(), 3e6);
        assert_eq!(Stress::gigapascals(1.0).in_megapascals(), 1e3);
        assert_eq!(Length::inches(1.0).in_millimeters(), 25.4);
        assert_eq!(Time::hours(1.0).in_minutes(), 6e1)
    }
    #[test]
    fn the_base_scale_reads_back_the_value_held() {
        assert_eq!(
            Stress::megapascals(3.0).in_pascals(),
            Stress::megapascals(3.0).value()
        )
    }
    //
    // An offset is what a division cannot undo, which is why a reader is named
    // rather than handed the scale to divide by.
    //
    #[test]
    fn an_offset_reads_back_through_its_own_reader() {
        assert_eq!(Temperature::celsius(20.0).in_celsius(), 20.0);
        assert_eq!(Temperature::celsius(20.0).in_kelvin(), 293.15);
        assert_eq!(Temperature::kelvin(273.15).in_celsius(), 0.0);
        assert_eq!(Temperature::fahrenheit(-40.0).in_celsius(), -40.0);
        assert_eq!(Temperature::celsius(100.0).in_fahrenheit(), 212.0)
    }
}
