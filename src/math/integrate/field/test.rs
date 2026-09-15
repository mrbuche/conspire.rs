use super::{Flat, Integrable, List, Product, Unimodular, integrate_euler, integrate_rkmk};
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank1, TensorRank2, TensorTuple, TensorVector,
    integrate::{Times, ode::explicit::variable_step::bogacki_shampine},
};
use crate::units::{Dimensionless, Rate, Time};

type BogackiShampine = bogacki_shampine::Tableau;

const RATE: Quantity<Rate> = Rate::per_second(1.0);

type Fp = TensorRank2<3, Current, Current, Dimensionless>;
type FpRate = TensorRank2<3, Current, Current, Rate>;

fn trace_free_rate() -> FpRate {
    FpRate::from([[0.1, 0.7, -0.3], [-0.4, 0.2, 0.5], [0.2, -0.1, -0.3]])
}

fn steps() -> Vec<Quantity<Time>> {
    (0..=20).map(|i| Quantity::new(0.05 * i as f64)).collect()
}

fn times() -> Vec<Quantity<Time>> {
    (0..=8).map(|i| Quantity::new(0.1 * i as f64)).collect()
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
    type Point = <Field as Integrable>::Point;
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
    assert!((last.1.1.determinant() - 1.0).abs() < 1e-10);
    assert!((last.0.value() - 1.5).abs() < 1e-12);
}

#[test]
fn list_field_advances_each_entry_by_its_own_flat_rate() {
    let time = times();
    let rates = [-RATE, RATE * 2.0, Quantity::<Rate>::new(0.0)];
    let starts: TensorVector<Quantity> =
        [Quantity::new(1.0), Quantity::new(1.0), Quantity::new(1.0)].into();
    let (_, points): (Times, TensorVector<TensorVector<Quantity>>) =
        integrate_euler::<List<Flat<Quantity>>, _, _>(
            |_: Quantity<Time>, y: &TensorVector<Quantity>| {
                Ok(y.iter()
                    .zip(rates.iter())
                    .map(|(y, rate)| y * rate)
                    .collect())
            },
            &time,
            starts,
        )
        .unwrap();
    let last = points.iter().last().unwrap();
    rates.iter().zip(last.iter()).for_each(|(rate, entry)| {
        let (_, reference): (Times, TensorVector<Quantity>) =
            integrate_euler::<Flat<Quantity>, _, _>(
                |_: Quantity<Time>, y: &Quantity| Ok(y * rate),
                &time,
                Quantity::new(1.0),
            )
            .unwrap();
        assert_eq!(entry.value(), reference.iter().last().unwrap().value());
    });
}

#[test]
fn list_field_keeps_every_entry_on_the_group_through_rkmk() {
    let rate_a = trace_free_rate();
    let rate_b = rate_a.clone() * -2.0;
    let starts: TensorVector<Fp> = [Fp::identity(), Fp::identity()].into();
    let (_, points): (Times, TensorVector<TensorVector<Fp>>) =
        integrate_rkmk::<List<Unimodular<Current>>, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &TensorVector<Fp>| Ok([rate_a.clone(), rate_b.clone()].into()),
            &steps(),
            starts,
        )
        .unwrap();
    let last = points.iter().last().unwrap();
    last.iter()
        .for_each(|fp| assert!((fp.determinant() - 1.0).abs() < 1e-10));
    let (_, reference_a): (Times, TensorVector<Fp>) =
        integrate_rkmk::<Unimodular<Current>, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &Fp| Ok(rate_a.clone()),
            &steps(),
            Fp::identity(),
        )
        .unwrap();
    last.iter()
        .next()
        .unwrap()
        .iter()
        .zip(reference_a.iter().last().unwrap().iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}
