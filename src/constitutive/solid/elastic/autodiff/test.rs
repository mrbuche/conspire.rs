//! Shared assertions and timing harness for the autodiff model cross-checks.

use crate::{
    constitutive::ConstitutiveError,
    math::{TensorRank2, TensorRank4},
    mechanics::{DeformationGradient, test::get_deformation_gradient},
    units::Stress,
};

pub(crate) fn ok<T>(result: Result<T, ConstitutiveError>) -> T {
    match result {
        Ok(value) => value,
        Err(_) => panic!("evaluation failed"),
    }
}

pub(crate) fn assert_close_2<I, J>(
    ad: &TensorRank2<3, I, J, Stress>,
    hand: &TensorRank2<3, I, J, Stress>,
    tol: f64,
) {
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = (ad[i][j].value(), hand[i][j].value());
            assert!(
                (a - b).abs() <= tol * (1.0 + b.abs()),
                "[{i}][{j}]: {a} vs {b}"
            );
        }
    }
}

pub(crate) fn assert_close_4<I, J, K, L>(
    ad: &TensorRank4<3, I, J, K, L, Stress>,
    hand: &TensorRank4<3, I, J, K, L, Stress>,
    tol: f64,
) {
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let (a, b) = (ad[i][j][k][l].value(), hand[i][j][k][l].value());
                    assert!(
                        (a - b).abs() <= tol * (1.0 + b.abs()),
                        "[{i}][{j}][{k}][{l}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}

// --- timing ---------------------------------------------------------------
//
// Each model's `time_*` tests are #[ignore]d, one path per test so their
// codegen does not interact. Inputs vary per iteration (256 distinct F) to
// defeat constant folding; the reported figure is the min over trials. Run
// serially:
//
//   cargo +nightly test --release --features autodiff --lib -j1 \
//     time_ -- --ignored --nocapture --test-threads=1

pub(crate) fn deformation_gradients() -> Vec<DeformationGradient> {
    let base = get_deformation_gradient();
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = base[i][j].value();
        }
    }
    (0..256)
        .map(|n| {
            let t = n as f64 * 0.013;
            DeformationGradient::from([
                [
                    b[0][0] + 0.05 * t.sin(),
                    b[0][1] + 0.04 * (t * 1.7).cos(),
                    b[0][2] - 0.03 * (t * 0.9).sin(),
                ],
                [
                    b[1][0] - 0.035 * (t * 1.3).sin(),
                    b[1][1] + 0.045 * (t * 0.7).cos(),
                    b[1][2] + 0.03 * (t * 1.1).sin(),
                ],
                [
                    b[2][0] + 0.03 * (t * 1.9).cos(),
                    b[2][1] - 0.04 * (t * 0.5).sin(),
                    b[2][2] + 0.05 * (t * 1.2).cos(),
                ],
            ])
        })
        .collect()
}

pub(crate) fn bench(
    label: &str,
    gradients: &[DeformationGradient],
    mut run: impl FnMut(&DeformationGradient),
) {
    use std::hint::black_box;
    use std::time::Instant;
    let n = gradients.len();
    for gradient in gradients {
        run(gradient);
    }
    let (trials, iters) = (25, 200_000);
    let mut best = f64::INFINITY;
    for _ in 0..trials {
        let start = Instant::now();
        for i in 0..iters {
            run(black_box(&gradients[i % n]));
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        best = best.min(per);
    }
    println!("{label:<26} {best:8.2} ns/call  (min of {trials})");
}

macro_rules! timing {
    ($name: ident, $label: expr, $model: expr, $call: expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let model = $model;
            let gradients =
                $crate::constitutive::solid::elastic::autodiff::test::deformation_gradients();
            $crate::constitutive::solid::elastic::autodiff::test::bench($label, &gradients, |f| {
                std::hint::black_box(($call)(&model, f));
            });
        }
    };
}
pub(crate) use timing;

/// All twelve `time_*` tests for one model: hand-written vs `Autodiff<M>` on
/// each of the six stress / tangent quantities. `$tag` labels the printed
/// lines; `ok` and `Elastic` must be in scope at the call site.
macro_rules! timing_all {
    ($tag: literal, $hand: expr, $autodiff: expr, $Hand: ty, $Ad: ty) => {
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_cauchy_stress,
            concat!($tag, " hand   sigma stress"),
            $hand,
            |m: &$Hand, f: &_| ok(m.cauchy_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_cauchy_stress,
            concat!($tag, " enzyme sigma stress"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.cauchy_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_first_piola_stress,
            concat!($tag, " hand   P stress"),
            $hand,
            |m: &$Hand, f: &_| ok(m.first_piola_kirchhoff_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_first_piola_stress,
            concat!($tag, " enzyme P stress"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.first_piola_kirchhoff_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_second_piola_stress,
            concat!($tag, " hand   S stress"),
            $hand,
            |m: &$Hand, f: &_| ok(m.second_piola_kirchhoff_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_second_piola_stress,
            concat!($tag, " enzyme S stress"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.second_piola_kirchhoff_stress(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_cauchy_tangent,
            concat!($tag, " hand   sigma tangent"),
            $hand,
            |m: &$Hand, f: &_| ok(m.cauchy_tangent_stiffness(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_cauchy_tangent,
            concat!($tag, " enzyme sigma tangent"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.cauchy_tangent_stiffness(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_first_piola_tangent,
            concat!($tag, " hand   P tangent"),
            $hand,
            |m: &$Hand, f: &_| ok(m.first_piola_kirchhoff_tangent_stiffness(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_first_piola_tangent,
            concat!($tag, " enzyme P tangent"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.first_piola_kirchhoff_tangent_stiffness(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_hand_second_piola_tangent,
            concat!($tag, " hand   S tangent"),
            $hand,
            |m: &$Hand, f: &_| ok(m.second_piola_kirchhoff_tangent_stiffness(f))
        );
        $crate::constitutive::solid::elastic::autodiff::test::timing!(
            time_enzyme_second_piola_tangent,
            concat!($tag, " enzyme S tangent"),
            $autodiff,
            |m: &$Ad, f: &_| ok(m.second_piola_kirchhoff_tangent_stiffness(f))
        );
    };
}
pub(crate) use timing_all;
