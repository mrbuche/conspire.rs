use super::HermiteSegment;
use crate::math::{
    Current, Quantity, Tensor, TensorArray, TensorRank2, TensorVec, TensorVector,
    integrate::{
        FreeInterpolant, Ode23, Times,
        field::{Flat, Integrable, Unimodular, integrate_rkmk_dae_adaptive},
        ode::explicit::variable_step::bogacki_shampine,
    },
};
use crate::units::{Dimensionless, Rate, Time};

type BogackiShampine = bogacki_shampine::Tableau;

type Fp = TensorRank2<3, Current, Current, Dimensionless>;
type FpRate = TensorRank2<3, Current, Current, Rate>;

fn constant_exponent() -> [[f64; 3]; 3] {
    [[0.0, 0.4, -0.2], [-0.3, 0.0, 0.5], [0.1, -0.15, 0.0]]
}

fn uniform_time(steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|i| Quantity::new(i as f64 / steps as f64))
        .collect()
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
