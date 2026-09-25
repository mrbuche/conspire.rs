use super::integrate_euler;
use crate::math::{
    Quantity, Tensor, TensorVector,
    integrate::{Euler, Explicit, Times, field::Flat},
};
use crate::units::{Rate, Time};

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
