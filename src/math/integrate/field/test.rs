use super::{Flat, Unimodular, integrate_euler};
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank2, TensorVector,
    integrate::{Euler, Explicit, Times},
};
use crate::units::{Dimensionless, Rate, Time};

const RATE: Quantity<Rate> = Rate::per_second(1.0);

fn times() -> Vec<Quantity<Time>> {
    (0..=8).map(|i| Quantity::new(0.1 * i as f64)).collect()
}

#[test]
fn flat_scalar_reproduces_euler_bit_for_bit() {
    let time = times();
    let (_, points): (Times, TensorVector<Quantity>) = integrate_euler::<Flat<Quantity>, _, _>(
        |_: Quantity<Time>, y: &Quantity| Ok(y * -RATE),
        &time,
        Quantity::new(1.0),
    )
    .unwrap();
    let (_, reference, _): (Times, TensorVector<Quantity>, TensorVector<Quantity<Rate>>) =
        Euler::default()
            .integrate(
                |_: Quantity<Time>, y: &Quantity| Ok(y * -RATE),
                &time,
                Quantity::new(1.0),
            )
            .unwrap();
    assert_eq!(points.iter().count(), time.len());
    points
        .iter()
        .zip(reference.iter())
        .for_each(|(a, b)| assert_eq!(a.value(), b.value()));
}

type Fp = TensorRank2<3, Current, Current, Dimensionless>;
type FpRate = TensorRank2<3, Current, Current, Rate>;

fn trace_free_symmetric_rate() -> FpRate {
    FpRate::from([[0.1, 0.2, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, -0.2]])
}

fn steps() -> Vec<Quantity<Time>> {
    (0..=20).map(|i| Quantity::new(0.05 * i as f64)).collect()
}

#[test]
fn unimodular_update_preserves_unit_determinant() {
    let rate = trace_free_symmetric_rate();
    let (_, points): (Times, TensorVector<Fp>) = integrate_euler::<Unimodular<Current>, _, _>(
        |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
        &steps(),
        Fp::identity(),
    )
    .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() < 1e-12);
}

#[test]
fn additive_update_of_the_same_rate_drifts_off_the_group() {
    let rate = trace_free_symmetric_rate();
    let (_, points): (Times, TensorVector<Fp>) = integrate_euler::<Flat<Fp>, _, _>(
        |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
        &steps(),
        Fp::identity(),
    )
    .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() > 1e-4);
}
