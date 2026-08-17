use super::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, GAS_CONSTANT, LIGHT_SPEED,
    ROOM_TEMPERATURE,
};
use crate::units::{Energy, Length, Time};

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
fn light_speed() {
    assert_eq!(
        (LIGHT_SPEED * Time::seconds(1.0)).in_meters(),
        Length::meters(2.997_924_58e8).in_meters()
    )
}
