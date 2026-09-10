use super::{Flat, IntegrableField, Product, Unimodular, integrate_euler};
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank1, TensorRank2, TensorTuple, TensorVector,
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

// trace 0, materially non-symmetric: exercises the scaling-and-squaring branch of `expm`
fn trace_free_rate() -> FpRate {
    FpRate::from([[0.1, 0.7, -0.3], [-0.4, 0.2, 0.5], [0.2, -0.1, -0.3]])
}

fn steps() -> Vec<Quantity<Time>> {
    (0..=20).map(|i| Quantity::new(0.05 * i as f64)).collect()
}

#[test]
fn unimodular_update_preserves_unit_determinant() {
    let rate = trace_free_rate();
    let (_, points): (Times, TensorVector<Fp>) = integrate_euler::<Unimodular<Current>, _, _>(
        |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
        &steps(),
        Fp::identity(),
    )
    .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() < 1e-10);
}

#[test]
fn additive_update_of_the_same_rate_drifts_off_the_group() {
    let rate = trace_free_rate();
    let (_, points): (Times, TensorVector<Fp>) = integrate_euler::<Flat<Fp>, _, _>(
        |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
        &steps(),
        Fp::identity(),
    )
    .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() > 1e-4);
}

#[test]
fn three_deep_product_field_round_trips_through_the_driver() {
    type Back = TensorRank1<3, Current>;
    type Field = Product<Flat<Quantity>, Product<Flat<Back>, Unimodular<Current>>>;
    type Point = <Field as IntegrableField>::Point;
    let dgamma_rate = Quantity::<Rate>::new(0.5);
    let back_rate = TensorRank1::<3, Current, Rate>::from([1.0, -2.0, 3.0]);
    let fp_rate = trace_free_rate();
    let (_, points): (Times, TensorVector<Point>) = integrate_euler::<Field, _, _>(
        |_: Quantity<Time>, _: &Point| {
            Ok(TensorTuple(
                dgamma_rate,
                TensorTuple(back_rate.clone(), fp_rate.clone()),
            ))
        },
        &steps(),
        TensorTuple(
            Quantity::new(1.0),
            TensorTuple(Back::zero(), Fp::identity()),
        ),
    )
    .unwrap();
    assert_eq!(points.iter().count(), steps().len());
    let last = points.iter().last().unwrap();
    // the F_p leaf kept det = 1 through the exponential-map updates
    assert!((last.1.1.determinant() - 1.0).abs() < 1e-10);
    // the scalar leaf advanced additively over a total time of 1.0
    assert!((last.0.value() - 1.5).abs() < 1e-12);
}
