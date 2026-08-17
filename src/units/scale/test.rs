use super::super::{
    Action, Amount, Area, Dimensionless, Energy, Entropy, Force, ForcePerLength, Length,
    MolarEnergy, MolarEntropy, Rate, ReciprocalAmount, Stress, StressPerLength, Temperature, Time,
    Velocity, Viscosity, Volume,
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
    fn a_velocity_over_a_time_is_a_length() {
        same(
            Velocity::meters_per_second(1.0) * Time::seconds(1.0),
            Length::meters(1.0),
        );
        same(
            Velocity::kilometers_per_hour(3.6),
            Velocity::meters_per_second(1.0),
        );
        same(
            Velocity::kilometers_per_hour(1.0) * Time::hours(1.0),
            Length::kilometers(1.0),
        )
    }
    #[test]
    fn an_energy_over_a_time_is_an_action() {
        same(
            Energy::joules(1.0) * Time::seconds(1.0),
            Action::joule_seconds(1.0),
        );
        same(
            Action::joule_seconds(1.0) * Rate::hertz(1.0),
            Energy::joules(1.0),
        )
    }
    #[test]
    fn an_amount_undoes_a_reciprocal_amount() {
        same(
            Amount::moles(1.0) * ReciprocalAmount::per_mole(1.0),
            Dimensionless::of(1.0),
        )
    }
    #[test]
    fn a_molar_quantity_is_one_over_an_amount() {
        same(
            Entropy::joules_per_kelvin(1.0) * ReciprocalAmount::per_mole(1.0),
            MolarEntropy::joules_per_mole_kelvin(1.0),
        );
        same(
            Energy::joules(1.0) * ReciprocalAmount::per_mole(1.0),
            MolarEnergy::joules_per_mole(1.0),
        );
        same(
            MolarEntropy::joules_per_mole_kelvin(1.0) * Temperature::kelvin(1.0),
            MolarEnergy::joules_per_mole(1.0),
        );
        same(
            MolarEnergy::kilojoules_per_mole(1.0) * Amount::moles(1.0),
            Energy::kilojoules(1.0),
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
    #[test]
    fn no_unit_at_all_reads_back_as_the_number_it_is() {
        assert_eq!(Dimensionless::of(0.3).value(), 0.3);
        assert_eq!(Dimensionless::percent(50.0).value(), 0.5);
        assert_eq!(Dimensionless::of(0.5).in_percent(), 50.0)
    }
    #[test]
    fn an_offset_reads_back_through_its_own_reader() {
        assert_eq!(Temperature::celsius(20.0).in_celsius(), 20.0);
        assert_eq!(Temperature::celsius(20.0).in_kelvin(), 293.15);
        assert_eq!(Temperature::kelvin(273.15).in_celsius(), 0.0);
        assert_eq!(Temperature::fahrenheit(-40.0).in_celsius(), -40.0);
        assert_eq!(Temperature::celsius(100.0).in_fahrenheit(), 212.0)
    }
}

mod named_in_words {
    use super::*;
    use crate::units::length_scale;

    fn is(label: &str, expected: Quantity<Length>) {
        match length_scale(label) {
            Some(scale) => same(scale(1.0), expected),
            None => panic!("{label} was not taken for a length"),
        }
    }

    #[test]
    fn a_length_is_known_by_symbol_or_by_name() {
        is("m", Length::meters(1.0));
        is("mm", Length::millimeters(1.0));
        is("millimeters", Length::millimeters(1.0));
        is("cm", Length::centimeters(1.0));
        is("um", Length::micrometers(1.0));
        is("microns", Length::micrometers(1.0));
        is("nm", Length::nanometers(1.0));
        is("km", Length::kilometers(1.0));
        is("in", Length::inches(1.0));
        is("feet", Length::feet(1.0))
    }

    #[test]
    fn a_length_is_known_however_it_is_spelled_or_spaced() {
        is("metres", Length::meters(1.0));
        is("Millimetres", Length::millimeters(1.0));
        is("  MM  ", Length::millimeters(1.0))
    }

    #[test]
    fn what_is_not_a_length_is_not_taken_for_one() {
        assert!(length_scale("radians").is_none());
        assert!(length_scale("seconds").is_none());
        assert!(length_scale("").is_none());
        assert!(length_scale("meters per second").is_none())
    }
}
