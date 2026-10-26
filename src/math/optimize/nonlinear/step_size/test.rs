use super::{BarzilaiBorwein, BarzilaiBorweinStep, Secant, StepEstimator};
use crate::{
    math::{Quantity, Scalar, Vector},
    units::{Length, ReciprocalLength, UnitDiv},
};

fn barzilai_borwein(step: BarzilaiBorweinStep) -> BarzilaiBorwein {
    BarzilaiBorwein { step }
}

const SS: Scalar = 6.0;
const SY: Scalar = 3.0;
const YY: Scalar = 2.0;

#[test]
fn short_is_sy_over_yy() {
    assert_eq!(
        barzilai_borwein(BarzilaiBorweinStep::Short).estimate(SS, SY, YY, 0),
        1.5
    )
}

#[test]
fn long_is_ss_over_sy() {
    assert_eq!(
        barzilai_borwein(BarzilaiBorweinStep::Long).estimate(SS, SY, YY, 0),
        2.0
    )
}

#[test]
fn alternating_takes_long_then_short() {
    let alternating = barzilai_borwein(BarzilaiBorweinStep::Alternating);
    assert_eq!(alternating.estimate(SS, SY, YY, 0), 2.0);
    assert_eq!(alternating.estimate(SS, SY, YY, 1), 1.5);
    assert_eq!(alternating.estimate(SS, SY, YY, 2), 2.0)
}

#[test]
fn the_short_step_is_never_longer_than_the_long_one() {
    assert!(
        barzilai_borwein(BarzilaiBorweinStep::Short).estimate(SS, SY, YY, 0)
            <= barzilai_borwein(BarzilaiBorweinStep::Long).estimate(SS, SY, YY, 0)
    )
}

mod secant {
    use super::*;
    const CURVATURE: Scalar = 4.0;
    fn residual(x: &Vector) -> Vector {
        x * CURVATURE
    }
    fn secant() -> (Secant<Vector, Vector>, Vector) {
        let start = Vector::from([1.0, -2.0]);
        (Secant::new(&start), start)
    }
    #[test]
    fn the_first_look_has_no_change_to_go_on() {
        let (mut secant, start) = secant();
        assert_eq!(
            secant
                .step_size(
                    &barzilai_borwein(BarzilaiBorweinStep::Short),
                    &start,
                    &residual(&start),
                    Quantity::new(0.5)
                )
                .value(),
            0.5
        )
    }
    #[test]
    fn a_quadratic_is_stepped_by_its_inverse_curvature() {
        for step in [
            BarzilaiBorweinStep::Short,
            BarzilaiBorweinStep::Long,
            BarzilaiBorweinStep::Alternating,
        ] {
            let (mut secant, start) = secant();
            let estimator = barzilai_borwein(step);
            secant.step_size(&estimator, &start, &residual(&start), Quantity::new(0.5));
            let next = Vector::from([0.25, -0.5]);
            let size = secant.step_size(&estimator, &next, &residual(&next), Quantity::new(0.5));
            assert!((size.value() - 1.0 / CURVATURE).abs() < 1e-12)
        }
    }
    #[test]
    fn the_last_solution_shown_is_kept() {
        let (mut secant, start) = secant();
        let next = Vector::from([0.25, -0.5]);
        secant.step_size(
            &barzilai_borwein(BarzilaiBorweinStep::Short),
            &start,
            &residual(&start),
            Quantity::new(0.5),
        );
        secant.step_size(
            &barzilai_borwein(BarzilaiBorweinStep::Short),
            &next,
            &residual(&next),
            Quantity::new(0.5),
        );
        assert_eq!(secant.solution(), &next)
    }
}

#[test]
fn a_step_takes_the_unit_of_the_solution_over_the_residual() {
    type Step = Quantity<<Length as UnitDiv<ReciprocalLength>>::Output>;
    let mut secant = Secant::new(&Quantity::<Length>::new(1.0));
    let estimator = barzilai_borwein(BarzilaiBorweinStep::Short);
    let residual = |x: &Quantity<Length>| Quantity::<ReciprocalLength>::new(4.0 * x.value());
    let start = Quantity::<Length>::new(1.0);
    secant.step_size(&estimator, &start, &residual(&start), Step::new(0.5));
    let next = Quantity::<Length>::new(0.25);
    let size: Step = secant.step_size(&estimator, &next, &residual(&next), Step::new(0.5));
    assert!((size.value() - 0.25).abs() < 1e-12)
}
