use super::{Conservative, ConservativeAt, Load, LoadAt, Loads};
use crate::{
    domain::boundary::{Field, Solid},
    math::Quantity,
    units::Time,
};

struct Pull(f64);

impl LoadAt<[f64; 2], [f64; 2]> for Pull {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        state: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        self.subtract_from(state, residual)
    }
}

impl Load<[f64; 2], [f64; 2]> for Pull {
    fn subtract_from(&self, _: &[f64; 2], residual: &mut [f64; 2]) -> Result<(), String> {
        residual[0] -= self.0;
        Ok(())
    }
}

impl ConservativeAt<[f64; 2], f64> for Pull {
    fn potential_energy_at(&self, _: Quantity<Time>, state: &[f64; 2]) -> Result<f64, String> {
        self.potential_energy(state)
    }
}

impl Conservative<[f64; 2], f64> for Pull {
    fn potential_energy(&self, state: &[f64; 2]) -> Result<f64, String> {
        Ok(-self.0 * state[0])
    }
}

struct Ramp(f64);

impl LoadAt<[f64; 2], [f64; 2]> for Ramp {
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        _: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        residual[1] -= self.0 * time.value();
        Ok(())
    }
}

struct Failing;

impl LoadAt<[f64; 2], [f64; 2]> for Failing {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        state: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        self.subtract_from(state, residual)
    }
}

impl Load<[f64; 2], [f64; 2]> for Failing {
    fn subtract_from(&self, _: &[f64; 2], _: &mut [f64; 2]) -> Result<(), String> {
        Err("failed".to_string())
    }
}

#[test]
fn no_loads_leave_the_residual_alone() {
    let mut residual = [1.0, 2.0];
    ().subtract_from(&[0.0; 2], &mut residual).unwrap();
    assert_eq!(residual, [1.0, 2.0]);
    assert_eq!(Conservative::potential_energy(&(), &[3.0; 2]), Ok(0.0));
}

#[test]
fn loads_add_up() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0)).with(Pull(3.0));
    let mut residual = [10.0, 0.0];
    loads.0.subtract_from(&[0.0; 2], &mut residual).unwrap();
    assert_eq!(residual, [5.0, 0.0]);
    assert_eq!(loads.0.potential_energy(&[4.0, 0.0]), Ok(-20.0));
}

#[test]
fn a_failing_load_fails_the_nest() {
    let loads = Loads::<Field<()>>::new().with(Pull(1.0)).with(Failing);
    assert_eq!(
        loads.0.subtract_from(&[0.0; 2], &mut [0.0; 2]),
        Err("failed".to_string())
    );
}

#[test]
fn loads_that_depend_on_time_take_the_time_they_are_given() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0)).with(Ramp(3.0));
    let mut residual = [10.0, 10.0];
    loads
        .0
        .subtract_at(Time::seconds(4.0), &[0.0; 2], &mut residual)
        .unwrap();
    assert_eq!(residual, [8.0, -2.0]);
}

#[test]
fn static_loads_ignore_the_time() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0));
    let (mut early, mut late) = ([10.0, 0.0], [10.0, 0.0]);
    loads
        .0
        .subtract_at(Time::seconds(0.0), &[0.0; 2], &mut early)
        .unwrap();
    loads
        .0
        .subtract_at(Time::seconds(9.0), &[0.0; 2], &mut late)
        .unwrap();
    assert_eq!(early, late);
}
