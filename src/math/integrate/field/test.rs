use super::{
    Flat, HermiteSegment, Integrable, List, Product, Unimodular, integrate_euler, integrate_rkmk,
    integrate_rkmk_adaptive, integrate_rkmk_dae_adaptive, rkmk_dae_step,
};
use crate::math::{
    Current, Derivative, Intermediate, Quantity, Reference, Tensor, TensorArray, TensorRank1,
    TensorRank2, TensorTuple, TensorVec, TensorVector,
    integrate::{
        BogackiShampine as LegacyBogackiShampine, Euler, Explicit, ExplicitDaeVariableStepExplicit,
        FreeInterpolant, Ode23, Times, ode::explicit::variable_step::bogacki_shampine,
    },
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

fn constant_exponent() -> [[f64; 3]; 3] {
    [[0.0, 0.4, -0.2], [-0.3, 0.0, 0.5], [0.1, -0.15, 0.0]]
}

fn uniform_time(steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|i| Quantity::new(i as f64 / steps as f64))
        .collect()
}

fn rkmk_endpoint_error(steps: usize) -> f64 {
    let a = FpRate::from(constant_exponent());
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| Ok(a.clone() * (1.0 / (1.0 + t.value()))),
            &uniform_time(steps),
            Fp::identity(),
        )
        .unwrap();
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
fn rkmk_reuses_the_fsal_stage_across_steps() {
    use std::cell::Cell;
    let a = FpRate::from(constant_exponent());
    let evaluations = Cell::new(0_usize);
    let steps = 8;
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk::<Unimodular<Current>, BogackiShampine, _, _>(
            |t: Quantity<Time>, _: &Fp| {
                evaluations.set(evaluations.get() + 1);
                Ok(a.clone() * (1.0 / (1.0 + t.value())))
            },
            &uniform_time(steps),
            Fp::identity(),
        )
        .unwrap();
    assert_eq!(evaluations.get(), 3 * steps + 1);
    let exact = (Fp::from(constant_exponent()) * 2.0_f64.ln())
        .expm()
        .unwrap();
    assert!((points.iter().last().unwrap() - &exact).norm().value() < 1e-2);
}

#[test]
fn rkmk_dae_reuses_the_fsal_stage_across_steps() {
    use std::cell::Cell;
    let rate = trace_free_rate();
    let rate_evaluations = Cell::new(0_usize);
    let solve_evaluations = Cell::new(0_usize);
    let steps = 8_usize;
    let dt = Quantity::<Time>::new(1.0 / steps as f64);
    let mut scratch = Vec::new();
    let mut point = Fp::identity();
    let mut z = Quantity::new(0.0);
    let mut carry = None;
    let mut t = Quantity::<Time>::new(0.0);
    for _ in 0..steps {
        let (advanced_point, advanced_z, next_carry) =
            rkmk_dae_step::<Unimodular<Current>, BogackiShampine, Quantity, Time>(
                &mut |_: Quantity<Time>, _: &Fp, _: &Quantity| {
                    rate_evaluations.set(rate_evaluations.get() + 1);
                    Ok(rate.clone())
                },
                &mut |t: Quantity<Time>, _: &Fp, _: &Quantity| {
                    solve_evaluations.set(solve_evaluations.get() + 1);
                    Ok(Quantity::new(t.value()))
                },
                &point,
                &z,
                t,
                dt,
                &mut scratch,
                carry.as_ref(),
            )
            .unwrap();
        point = advanced_point;
        z = advanced_z;
        carry = next_carry;
        t += dt;
    }
    assert_eq!(rate_evaluations.get(), 3 * steps + 1);
    assert_eq!(solve_evaluations.get(), 4 * steps + 1);
    assert!((point.determinant() - 1.0).abs() < 1e-10);
    assert!((z.value() - t.value()).abs() < 1e-12);
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

#[test]
fn rkmk_on_a_field_whose_increment_is_not_its_point() {
    type PlasticField = Unimodular<Intermediate, Reference>;
    type Fp = TensorRank2<3, Intermediate, Reference, Dimensionless>;
    type Dp = TensorRank2<3, Intermediate, Intermediate, Rate>;
    let d_p = Dp::from([[0.0, 0.4, -0.2], [-0.3, 0.0, 0.5], [0.1, -0.15, 0.0]]);
    let (_, points): (Times, TensorVector<Fp>) =
        integrate_rkmk::<PlasticField, BogackiShampine, _, _>(
            |_: Quantity<Time>, _: &Fp| Ok(d_p.clone()),
            &uniform_time(16),
            Fp::identity(),
        )
        .unwrap();
    let last = points.iter().last().unwrap();
    assert!((last.determinant() - 1.0).abs() < 1e-10);
    assert!((last - &Fp::identity()).norm().value() > 1e-2);
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

fn hermite_grid(points: usize) -> Vec<f64> {
    (0..=points).map(|i| i as f64 / points as f64).collect()
}

#[test]
fn hermite_on_a_flat_field_is_the_free_interpolant() {
    let h = Quantity::<Time>::new(0.3);
    let t_0 = Quantity::<Time>::new(0.7);
    let (y_0, y_1) = (Quantity::new(1.2), Quantity::new(0.4));
    let (v_0, v_1) = (Rate::per_second(-2.5), Rate::per_second(-0.9));
    let segment =
        HermiteSegment::<Flat<Quantity>, Time>::new(t_0, h, y_0, y_1 - y_0, v_0 * h, v_1 * h);
    let time = Times::from(
        hermite_grid(7)
            .iter()
            .map(|theta| t_0 + h * *theta)
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let mut tp = Times::new();
    tp.push(t_0);
    tp.push(t_0 + h);
    let mut yp = TensorVector::new();
    yp.push(y_0);
    yp.push(y_1);
    let mut dydtp = TensorVector::new();
    dydtp.push(v_0);
    dydtp.push(v_1);
    let (reference, _): (TensorVector<Quantity>, TensorVector<Quantity<Rate>>) =
        <Ode23 as FreeInterpolant<_, _, _, Time>>::interpolate_free(&time, &tp, &yp, &dydtp);
    time.iter()
        .zip(reference.iter())
        .for_each(|(time_k, reference_k)| {
            let error = (segment.evaluate(*time_k).unwrap() - reference_k)
                .value()
                .abs();
            assert!(error < 1e-15, "flat reduction differs by {error}");
        });
}

fn hermite_analytic_segment(t_0: f64, h: f64) -> HermiteSegment<Unimodular<Current>, Time> {
    let a = Fp::from(constant_exponent());
    let rate = FpRate::from(constant_exponent());
    HermiteSegment::new(
        Quantity::new(t_0),
        Quantity::new(h),
        (&a * (1.0 + t_0).ln()).expm().unwrap(),
        &a * ((1.0 + t_0 + h) / (1.0 + t_0)).ln(),
        &rate * Quantity::<Time>::new(h / (1.0 + t_0)),
        &rate * Quantity::<Time>::new(h / (1.0 + t_0 + h)),
    )
}

fn hermite_analytic_error(h: f64) -> f64 {
    let t_0 = 0.4;
    let a = Fp::from(constant_exponent());
    let segment = hermite_analytic_segment(t_0, h);
    hermite_grid(16)
        .iter()
        .map(|theta| {
            let t = t_0 + h * *theta;
            let exact = (&a * (1.0 + t).ln()).expm().unwrap();
            (segment.evaluate(Quantity::new(t)).unwrap() - &exact)
                .norm()
                .value()
        })
        .fold(0.0, f64::max)
}

#[test]
fn hermite_reproduces_both_endpoints_exactly() {
    let (t_0, h) = (0.4, 0.25);
    let segment = hermite_analytic_segment(t_0, h);
    let a = Fp::from(constant_exponent());
    let left = (&a * (1.0 + t_0).ln()).expm().unwrap();
    let displacement = &a * ((1.0 + t_0 + h) / (1.0 + t_0)).ln();
    let right = <Unimodular<Current> as Integrable>::reconstruct(&left, &displacement).unwrap();
    assert_eq!(
        (segment.evaluate(Quantity::new(t_0)).unwrap() - &left)
            .norm()
            .value(),
        0.0
    );
    assert_eq!(
        (segment.evaluate(Quantity::new(t_0 + h)).unwrap() - &right)
            .norm()
            .value(),
        0.0
    );
}

#[test]
fn hermite_interpolation_is_fourth_order() {
    let coarse = hermite_analytic_error(0.2);
    let fine = hermite_analytic_error(0.1);
    assert!(
        (1e-14..1e-2).contains(&coarse),
        "vacuous or diverged: {coarse}"
    );
    assert!(
        fine < coarse / 13.0,
        "halving the step should cut the error ~16x: {coarse} -> {fine}"
    );
}

#[test]
fn hermite_dense_output_keeps_the_group_state_unimodular() {
    let rate = FpRate::from(constant_exponent());
    let time = uniform_time(9);
    let (times, points, _): (Times, TensorVector<Fp>, TensorVector<Quantity>) =
        integrate_rkmk_dae_adaptive::<Unimodular<Current>, BogackiShampine, _, _, _, _>(
            |t: Quantity<Time>, _: &Fp, _: &Quantity| Ok(&rate * (1.0 / (1.0 + t.value()))),
            |_: Quantity<Time>, point: &Fp, _: &Quantity| Ok(Quantity::new(point.norm().value())),
            &time,
            (Fp::identity(), Quantity::new(3.0_f64.sqrt())),
            1e-6,
            0.0,
        )
        .unwrap();
    assert_eq!(times.iter().count(), time.len());
    points.iter().for_each(|point| {
        assert!((point.determinant() - 1.0).abs() < 1e-10);
    });
    let exact = (Fp::from(constant_exponent()) * 2.0_f64.ln())
        .expm()
        .unwrap();
    let error = (points.iter().last().unwrap() - &exact).norm().value();
    assert!(error < 1e-5, "dense endpoint error {error}");
}
