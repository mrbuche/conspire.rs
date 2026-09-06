use super::AutodiffNeoHookean;
use crate::{
    constitutive::solid::{
        elastic::{
            Elastic,
            autodiff::{
                Autodiff,
                test::{assert_close_2, assert_close_4, ok},
            },
        },
        hyperelastic::{Hyperelastic, NeoHookean},
    },
    mechanics::{DeformationGradient, test::get_deformation_gradient},
    units::Stress,
};

fn hand() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    }
}

fn autodiff() -> Autodiff<AutodiffNeoHookean> {
    Autodiff(AutodiffNeoHookean {
        bulk_modulus: Stress::pascals(1.3),
        shear_modulus: Stress::pascals(0.7),
    })
}

#[test]
fn matches_hand_written() {
    let (ad, hand, f) = (autodiff(), hand(), get_deformation_gradient());
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
//     time_ -- --ignored --nocapture --test-threads=1

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
    ($name: ident, $label: expr, $model: expr, $call: expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let model = $model;
            let gradients = deformation_gradients();
            bench($label, &gradients, |f| {
                std::hint::black_box(($call)(&model, f));
            });
        }
    };
}

type Ad = Autodiff<AutodiffNeoHookean>;

timing!(
    time_hand_cauchy_stress,
    "hand    sigma stress",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.cauchy_stress(f))
);
timing!(
    time_enzyme_cauchy_stress,
    "enzyme  sigma stress",
    autodiff(),
    |m: &Ad, f: &_| ok(m.cauchy_stress(f))
);
timing!(
    time_hand_first_piola_stress,
    "hand    P stress",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.first_piola_kirchhoff_stress(f))
);
timing!(
    time_enzyme_first_piola_stress,
    "enzyme  P stress",
    autodiff(),
    |m: &Ad, f: &_| ok(m.first_piola_kirchhoff_stress(f))
);
timing!(
    time_hand_second_piola_stress,
    "hand    S stress",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.second_piola_kirchhoff_stress(f))
);
timing!(
    time_enzyme_second_piola_stress,
    "enzyme  S stress",
    autodiff(),
    |m: &Ad, f: &_| ok(m.second_piola_kirchhoff_stress(f))
);
timing!(
    time_hand_cauchy_tangent,
    "hand    sigma tangent",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.cauchy_tangent_stiffness(f))
);
timing!(
    time_enzyme_cauchy_tangent,
    "enzyme  sigma tangent",
    autodiff(),
    |m: &Ad, f: &_| ok(m.cauchy_tangent_stiffness(f))
);
timing!(
    time_hand_first_piola_tangent,
    "hand    P tangent",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.first_piola_kirchhoff_tangent_stiffness(f))
);
timing!(
    time_enzyme_first_piola_tangent,
    "enzyme  P tangent",
    autodiff(),
    |m: &Ad, f: &_| ok(m.first_piola_kirchhoff_tangent_stiffness(f))
);
timing!(
    time_hand_second_piola_tangent,
    "hand    S tangent",
    hand(),
    |m: &NeoHookean, f: &_| ok(m.second_piola_kirchhoff_tangent_stiffness(f))
);
timing!(
    time_enzyme_second_piola_tangent,
    "enzyme  S tangent",
    autodiff(),
    |m: &Ad, f: &_| ok(m.second_piola_kirchhoff_tangent_stiffness(f))
);
