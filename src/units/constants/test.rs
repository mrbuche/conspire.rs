use super::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, GAS_CONSTANT, LIGHT_SPEED,
    PLANCK_CONSTANT, ROOM_TEMPERATURE, STANDARD_GRAVITY, STEFAN_BOLTZMANN_CONSTANT,
    VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY,
};
use crate::units::{ElectricPotential, Energy, Length, Time};

#[test]
fn gas_constant() {
    assert_eq!(AVOGADRO_CONSTANT * BOLTZMANN_CONSTANT, GAS_CONSTANT)
}

#[test]
fn room_temperature() {
    assert_eq!(ROOM_TEMPERATURE.in_celsius(), 20.0);
    assert_eq!(ROOM_TEMPERATURE.in_kelvin(), 293.15)
}

#[test]
fn electronvolt() {
    assert_eq!(
        Energy::electronvolts(1.0).in_joules(),
        ELEMENTARY_CHARGE.in_coulombs()
    )
}

#[test]
fn elementary_charge_through_a_volt_is_an_electronvolt() {
    assert_eq!(
        (ELEMENTARY_CHARGE * ElectricPotential::volts(1.0)).in_electronvolts(),
        1.0
    )
}

#[test]
fn vacuum_constants_give_the_light_speed() {
    let speed = LIGHT_SPEED.in_meters_per_second();
    assert!(
        (VACUUM_PERMITTIVITY.in_farads_per_meter()
            * VACUUM_PERMEABILITY.in_henries_per_meter()
            * speed
            * speed
            - 1.0)
            .abs()
            < 1e-9
    )
}

#[test]
fn light_speed() {
    assert_eq!(
        (LIGHT_SPEED * Time::seconds(1.0)).in_meters(),
        Length::meters(2.997_924_58e8).in_meters()
    )
}

#[test]
fn standard_gravity() {
    assert_eq!(STANDARD_GRAVITY.in_meters_per_second_squared(), 9.806_65)
}

#[test]
fn stefan_boltzmann_constant() {
    let (k, h, c) = (
        BOLTZMANN_CONSTANT.value(),
        PLANCK_CONSTANT.value(),
        LIGHT_SPEED.value(),
    );
    let expected = 2.0 * std::f64::consts::PI.powi(5) * k.powi(4) / (15.0 * h.powi(3) * c * c);
    assert!((STEFAN_BOLTZMANN_CONSTANT.value() / expected - 1.0).abs() < 1e-8)
}
