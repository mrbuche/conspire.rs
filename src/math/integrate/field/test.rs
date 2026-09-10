use super::{
    Flat, IntegrableField, Product, Unimodular, integrate_euler, integrate_rkmk,
    integrate_rkmk_adaptive,
};
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank1, TensorRank2, TensorTuple, TensorVector,
    integrate::{Euler, Explicit, Times, ode::explicit::variable_step::bogacki_shampine},
};
use crate::units::{Dimensionless, Rate, Time};

type BogackiShampine = bogacki_shampine::Tableau;

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

fn constant_exponent() -> [[f64; 3]; 3] {
    // trace 0, materially non-symmetric
    [[0.0, 0.4, -0.2], [-0.3, 0.0, 0.5], [0.1, -0.15, 0.0]]
}

fn uniform_time(steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|i| Quantity::new(i as f64 / steps as f64))
        .collect()
}

// X' = g(t) A X with a fixed A and a non-polynomial scalar g; the exact solution
// is exp(A ∫g) X0, and RKMK's error is the tableau's quadrature error on ∫g.
fn rkmk_endpoint_error(steps: usize) -> f64 {
    let a = FpRate::from(constant_exponent());
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| Ok(a.clone() * (1.0 / (1.0 + t.value()))),
            &uniform_time(steps),
            Fp::identity(),
        )
        .unwrap();
    // ∫_0^1 1/(1+t) dt = ln 2
    let exact = (Fp::from(constant_exponent()) * 2.0_f64.ln())
        .expm()
        .unwrap();
    (points.iter().last().unwrap() - &exact).norm().value()
}

#[test]
fn rkmk_matches_the_exact_exponential_for_a_constant_rate() {
    assert!(rkmk_endpoint_error(64) < 1e-3);
}

#[test]
fn rkmk_bogacki_shampine_is_third_order() {
    let coarse = rkmk_endpoint_error(8);
    let fine = rkmk_endpoint_error(16);
    // genuine truncation error, not the machine-precision floor
    assert!(
        (1e-9..1e-1).contains(&coarse),
        "vacuous or diverged: {coarse}"
    );
    assert!(
        fine < coarse / 6.0,
        "halving the step should cut the error ~8x: {coarse} -> {fine}"
    );
}

#[test]
fn rkmk_keeps_the_group_state_unimodular() {
    let rate = trace_free_rate();
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk::<Unimodular<Current>, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
            &steps(),
            Fp::identity(),
        )
        .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() < 1e-10);
}

#[test]
fn rkmk_on_a_flat_field_is_the_plain_tableau() {
    let time = uniform_time(10);
    let (_, rkmk): (Times, TensorVector<Quantity>) =
        integrate_rkmk::<Flat<Quantity>, BogackiShampine, _, _>(
            |_: Quantity<Time>, y: &Quantity| Ok(y * -RATE),
            &time,
            Quantity::new(1.0),
        )
        .unwrap();
    assert!((rkmk.iter().last().unwrap().value() - (-1.0_f64).exp()).abs() < 1e-4);
}

// span [0, 1], f = A/(1+t), exact endpoint exp(A ln 2); returns (accepted steps, endpoint error)
fn rkmk_adaptive_run(abs_tol: f64) -> (usize, f64) {
    let a = FpRate::from(constant_exponent());
    let (times, points): (Times, TensorVector<Fp>) =
        integrate_rkmk_adaptive::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| Ok(a.clone() * (1.0 / (1.0 + t.value()))),
            &uniform_time(1),
            Fp::identity(),
            abs_tol,
            0.0,
        )
        .unwrap();
    let exact = (Fp::from(constant_exponent()) * 2.0_f64.ln())
        .expm()
        .unwrap();
    (
        times.iter().count(),
        (points.iter().last().unwrap() - &exact).norm().value(),
    )
}

#[test]
fn rkmk_adaptive_meets_the_requested_tolerance() {
    let (steps, error) = rkmk_adaptive_run(1e-7);
    assert!(
        steps > 2,
        "the controller never sub-divided the span: {steps}"
    );
    assert!(
        error < 1e-5,
        "endpoint error {error} misses the requested tolerance"
    );
}

#[test]
fn rkmk_adaptive_subdivides_more_for_a_tighter_tolerance() {
    let (loose, loose_error) = rkmk_adaptive_run(1e-4);
    let (tight, tight_error) = rkmk_adaptive_run(1e-9);
    assert!(
        loose < tight && tight < 500,
        "step counts are not monotone and sane: {loose} -> {tight}"
    );
    assert!(
        tight_error < loose_error,
        "a tighter tolerance was no more accurate: {loose_error} -> {tight_error}"
    );
}

#[test]
fn rkmk_adaptive_keeps_the_group_state_unimodular() {
    let rate = trace_free_rate();
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk_adaptive::<Unimodular<Current>, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &Fp| Ok(rate.clone()),
            &uniform_time(1),
            Fp::identity(),
            1e-8,
            0.0,
        )
        .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() < 1e-10);
}
