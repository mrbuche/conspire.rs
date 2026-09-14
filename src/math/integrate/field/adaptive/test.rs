use super::{integrate_rkmk_adaptive, integrate_rkmk_dae_adaptive};
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank2, TensorVector,
    integrate::{Times, field::Unimodular, ode::explicit::variable_step::bogacki_shampine},
};
use crate::units::{Dimensionless, Rate, Time};

type BogackiShampine = bogacki_shampine::Tableau;

type Fp = TensorRank2<3, Current, Current, Dimensionless>;
type FpRate = TensorRank2<3, Current, Current, Rate>;

fn trace_free_rate() -> FpRate {
    FpRate::from([[0.1, 0.7, -0.3], [-0.4, 0.2, 0.5], [0.2, -0.1, -0.3]])
}

fn constant_exponent() -> [[f64; 3]; 3] {
    [[0.0, 0.4, -0.2], [-0.3, 0.0, 0.5], [0.1, -0.15, 0.0]]
}

fn uniform_time(steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|i| Quantity::new(i as f64 / steps as f64))
        .collect()
}

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

#[test]
fn rkmk_adaptive_retries_with_a_smaller_step_after_a_rate_failure() {
    use std::cell::Cell;
    let rate = trace_free_rate();
    let calls = Cell::new(0_usize);
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk_adaptive::<Unimodular<Current>, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &Fp| {
                calls.set(calls.get() + 1);
                if calls.get() <= 2 {
                    Err("simulated divergence: trial inverted an element".to_string())
                } else {
                    Ok(rate.clone())
                }
            },
            &uniform_time(1),
            Fp::identity(),
            1e-6,
            0.0,
        )
        .unwrap();
    assert!((points.iter().last().unwrap().determinant() - 1.0).abs() < 1e-10);
}

#[test]
fn rkmk_adaptive_errors_when_the_requested_tolerance_is_unreachable() {
    let a = FpRate::from(constant_exponent());
    let result: Result<(Times, TensorVector<Fp>), _> =
        integrate_rkmk_adaptive::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| Ok(a.clone() * (1.0 / (1.0 + t.value()))),
            &uniform_time(1),
            Fp::identity(),
            0.0,
            0.0,
        );
    assert!(
        result.is_err(),
        "an unreachable tolerance should error instead of silently accepting a step below the floor"
    );
}

#[test]
fn rkmk_dae_adaptive_errors_when_the_requested_tolerance_is_unreachable() {
    let rate = FpRate::from(constant_exponent());
    let result: Result<(Times, TensorVector<Fp>, TensorVector<Quantity>), _> =
        integrate_rkmk_dae_adaptive::<Unimodular<Current>, BogackiShampine, _, _, _, _>(
            |t: Quantity<Time>, _: &Fp, _: &Quantity| Ok(&rate * (1.0 / (1.0 + t.value()))),
            |_: Quantity<Time>, point: &Fp, _: &Quantity| Ok(Quantity::new(point.norm().value())),
            &uniform_time(1),
            (Fp::identity(), Quantity::new(3.0_f64.sqrt())),
            0.0,
            0.0,
        );
    assert!(
        result.is_err(),
        "an unreachable tolerance should error instead of silently accepting a step below the floor"
    );
}

#[test]
fn rkmk_adaptive_reports_on_the_group_at_requested_times() {
    let a = FpRate::from(constant_exponent());
    let requested: Vec<Quantity<Time>> = (0..=10).map(|i| Quantity::new(0.1 * i as f64)).collect();
    let (reported, points): (Times, TensorVector<Fp>) =
        integrate_rkmk_adaptive::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| Ok(a.clone() * (1.0 / (1.0 + t.value()))),
            &requested,
            Fp::identity(),
            1e-10,
            0.0,
        )
        .unwrap();
    assert_eq!(reported.iter().count(), requested.len());
    reported
        .iter()
        .zip(requested.iter())
        .for_each(|(a, b)| assert_eq!(a, b));
    requested
        .iter()
        .zip(points.iter())
        .skip(1)
        .for_each(|(t, point)| {
            let exact = (Fp::from(constant_exponent()) * (1.0 + t.value()).ln())
                .expm()
                .unwrap();
            assert!((point - &exact).norm().value() < 1e-3);
            assert!((point.determinant() - 1.0).abs() < 1e-10);
        });
}
