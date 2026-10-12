use super::{Objective, Settings, minimize};
use crate::math::Scalar;

const SETTINGS: Settings = Settings {
    backtracks: 40,
    convergence: 1.0e-12,
    history: 10,
    iterations: 500,
};

struct Quadratic(Vec<Scalar>);

impl Objective for Quadratic {
    fn value(&mut self, x: &[Scalar]) -> Scalar {
        x.iter().zip(&self.0).map(|(x, a)| 0.5 * a * x * x).sum()
    }
    fn gradient(&mut self, x: &[Scalar], gradient: &mut [Scalar]) {
        gradient
            .iter_mut()
            .zip(x.iter().zip(&self.0))
            .for_each(|(g, (x, a))| *g = a * x)
    }
}

struct Rosenbrock;

impl Objective for Rosenbrock {
    fn value(&mut self, x: &[Scalar]) -> Scalar {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }
    fn gradient(&mut self, x: &[Scalar], gradient: &mut [Scalar]) {
        gradient[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        gradient[1] = 200.0 * (x[1] - x[0] * x[0]);
    }
}

#[test]
fn solves_an_ill_conditioned_quadratic() {
    let scales: Vec<Scalar> = (0..20).map(|i| 10.0_f64.powf(i as Scalar / 5.0)).collect();
    let mut x = vec![1.0; 20];
    let outcome = minimize(&mut Quadratic(scales), &mut x, 1.0, &SETTINGS);
    assert!(outcome.value < 1.0e-12, "{}", outcome.value);
    assert!(outcome.value < outcome.initial);
}

#[test]
fn solves_rosenbrock() {
    let mut x = vec![-1.2, 1.0];
    minimize(&mut Rosenbrock, &mut x, 1.0, &SETTINGS);
    assert!(
        (x[0] - 1.0).abs() < 1.0e-6 && (x[1] - 1.0).abs() < 1.0e-6,
        "{x:?}"
    );
}

#[test]
fn settles_immediately_at_a_minimum() {
    let mut x = vec![0.0; 3];
    let outcome = minimize(&mut Quadratic(vec![1.0, 2.0, 3.0]), &mut x, 1.0, &SETTINGS);
    assert!(outcome.settled);
    assert_eq!(x, vec![0.0; 3]);
}

#[test]
fn never_raises_the_objective_when_the_budget_is_short() {
    let mut x = vec![-1.2, 1.0];
    let settings = Settings {
        iterations: 3,
        ..SETTINGS
    };
    let outcome = minimize(&mut Rosenbrock, &mut x, 1.0, &settings);
    assert!(outcome.value <= outcome.initial);
    assert!(!outcome.settled);
}
