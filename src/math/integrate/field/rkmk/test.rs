use super::rkmk_dae_step;
use crate::math::{
    Current, Intermediate, Quantity, Reference, Tensor, TensorArray, TensorRank2, TensorVector,
    integrate::{
        Times,
        field::{Unimodular, integrate_rkmk},
        ode::explicit::variable_step::bogacki_shampine,
    },
};
use crate::units::{Dimensionless, Rate, Time};

type BogackiShampine = bogacki_shampine::Tableau;

type Fp = TensorRank2<3, Current, Current, Dimensionless>;
type FpRate = TensorRank2<3, Current, Current, Rate>;

fn trace_free_rate() -> FpRate {
    FpRate::from([[0.1, 0.7, -0.3], [-0.4, 0.2, 0.5], [0.2, -0.1, -0.3]])
}

fn steps() -> Vec<Quantity<Time>> {
    (0..=20).map(|i| Quantity::new(0.05 * i as f64)).collect()
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
    use crate::math::integrate::field::Flat;
    const RATE: Quantity<Rate> = Rate::per_second(1.0);
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
