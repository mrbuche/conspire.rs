use crate::{
    constitutive::{
        ConstitutiveError,
        autodiff::neo_hookean,
        solid::{elastic::Elastic, hyperelastic::NeoHookean},
    },
    math::{TensorRank2, TensorRank4},
    mechanics::{DeformationGradient, test::get_deformation_gradient},
    units::Stress,
};

fn model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    }
}

fn ok<T>(result: Result<T, ConstitutiveError>) -> T {
    match result {
        Ok(value) => value,
        Err(_) => panic!("hand-written evaluation failed"),
    }
}

fn assert_close_2<I, J>(
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

fn assert_close_4<I, J, K, L>(
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

#[test]
fn stresses_match_hand_written() {
    let (model, f) = (model(), get_deformation_gradient());
    assert_close_2(
        &neo_hookean::cauchy_stress(&model, &f),
        &ok(model.cauchy_stress(&f)),
        1e-8,
    );
    assert_close_2(
        &neo_hookean::first_piola_kirchhoff_stress(&model, &f),
        &ok(model.first_piola_kirchhoff_stress(&f)),
        1e-8,
    );
    assert_close_2(
        &neo_hookean::second_piola_kirchhoff_stress(&model, &f),
        &ok(model.second_piola_kirchhoff_stress(&f)),
        1e-8,
    );
}

#[test]
fn tangents_match_hand_written() {
    let (model, f) = (model(), get_deformation_gradient());
    assert_close_4(
        &neo_hookean::cauchy_tangent_stiffness(&model, &f),
        &ok(model.cauchy_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &neo_hookean::first_piola_kirchhoff_tangent_stiffness(&model, &f),
        &ok(model.first_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &neo_hookean::second_piola_kirchhoff_tangent_stiffness(&model, &f),
        &ok(model.second_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
}

#[test]
fn wrapper_matches_hand_written() {
    use crate::constitutive::{
        autodiff::{Autodiff, neo_hookean::AutodiffNeoHookean},
        solid::hyperelastic::Hyperelastic,
    };
    let hand = model();
    let ad = Autodiff(AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    });
    let f = get_deformation_gradient();
    assert_close_2(&ok(ad.cauchy_stress(&f)), &ok(hand.cauchy_stress(&f)), 1e-8);
    assert_close_2(
        &ok(ad.first_piola_kirchhoff_stress(&f)),
        &ok(hand.first_piola_kirchhoff_stress(&f)),
        1e-8,
    );
    assert_close_2(
        &ok(ad.second_piola_kirchhoff_stress(&f)),
        &ok(hand.second_piola_kirchhoff_stress(&f)),
        1e-8,
    );
    assert_close_4(
        &ok(ad.cauchy_tangent_stiffness(&f)),
        &ok(hand.cauchy_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &ok(ad.first_piola_kirchhoff_tangent_stiffness(&f)),
        &ok(hand.first_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
    assert_close_4(
        &ok(ad.second_piola_kirchhoff_tangent_stiffness(&f)),
        &ok(hand.second_piola_kirchhoff_tangent_stiffness(&f)),
        1e-6,
    );
    let (energy_ad, energy_hand) = (
        ok(ad.helmholtz_free_energy_density(&f)).value(),
        ok(hand.helmholtz_free_energy_density(&f)).value(),
    );
    assert!((energy_ad - energy_hand).abs() <= 1e-8 * (1.0 + energy_hand.abs()));
}

// --- timing -----------------------------------------------------------------
//
// Each path is its own #[ignore]d test so their codegen does not interact.
// Inputs vary per iteration (256 distinct F) to defeat constant folding, and
// the reported figure is the min over many trials. Run serially:
//
//   cargo +nightly test --release --features autodiff --lib -j1 \
//     autodiff::test::time_ -- --ignored --nocapture --test-threads=1

fn deformation_gradients() -> Vec<DeformationGradient> {
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

fn bench(
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
    ($name: ident, $label: expr, $run: expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let model = model();
            let gradients = deformation_gradients();
            bench($label, &gradients, |f| {
                std::hint::black_box(($run)(&model, f));
            });
        }
    };
}

timing!(
    time_hand_cauchy_stress,
    "hand    sigma stress",
    |m: &NeoHookean, f: &_| ok(m.cauchy_stress(f))
);
timing!(
    time_enzyme_cauchy_stress,
    "enzyme  sigma stress",
    |m: &NeoHookean, f: &_| neo_hookean::cauchy_stress(m, f)
);
timing!(
    time_hand_first_piola_stress,
    "hand    P stress",
    |m: &NeoHookean, f: &_| ok(m.first_piola_kirchhoff_stress(f))
);
timing!(
    time_enzyme_first_piola_stress,
    "enzyme  P stress",
    |m: &NeoHookean, f: &_| neo_hookean::first_piola_kirchhoff_stress(m, f)
);
timing!(
    time_hand_second_piola_stress,
    "hand    S stress",
    |m: &NeoHookean, f: &_| ok(m.second_piola_kirchhoff_stress(f))
);
timing!(
    time_enzyme_second_piola_stress,
    "enzyme  S stress",
    |m: &NeoHookean, f: &_| neo_hookean::second_piola_kirchhoff_stress(m, f)
);
timing!(
    time_hand_cauchy_tangent,
    "hand    sigma tangent",
    |m: &NeoHookean, f: &_| ok(m.cauchy_tangent_stiffness(f))
);
timing!(
    time_enzyme_cauchy_tangent,
    "enzyme  sigma tangent",
    |m: &NeoHookean, f: &_| neo_hookean::cauchy_tangent_stiffness(m, f)
);
timing!(
    time_hand_first_piola_tangent,
    "hand    P tangent",
    |m: &NeoHookean, f: &_| ok(m.first_piola_kirchhoff_tangent_stiffness(f))
);
timing!(
    time_enzyme_first_piola_tangent,
    "enzyme  P tangent",
    |m: &NeoHookean, f: &_| neo_hookean::first_piola_kirchhoff_tangent_stiffness(m, f)
);
timing!(
    time_hand_second_piola_tangent,
    "hand    S tangent",
    |m: &NeoHookean, f: &_| ok(m.second_piola_kirchhoff_tangent_stiffness(f))
);
timing!(
    time_enzyme_second_piola_tangent,
    "enzyme  S tangent",
    |m: &NeoHookean, f: &_| neo_hookean::second_piola_kirchhoff_tangent_stiffness(m, f)
);
