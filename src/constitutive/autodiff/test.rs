use crate::{
    constitutive::{
        ConstitutiveError,
        autodiff::neo_hookean,
        solid::{elastic::Elastic, hyperelastic::NeoHookean},
    },
    math::{TensorRank2, TensorRank4},
    mechanics::test::get_deformation_gradient,
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

// Rough back-to-back timing, hand-written vs Enzyme. Run with:
//   cargo +nightly test --release --features autodiff --lib -j1 \
//     neo_hookean_timing -- --nocapture --ignored
#[test]
#[ignore]
fn neo_hookean_timing() {
    use std::hint::black_box;
    use std::time::Instant;
    let (model, f) = (model(), get_deformation_gradient());
    let iters = 1_000_000;
    let time = |label: &str, mut run: Box<dyn FnMut()>| {
        for _ in 0..iters / 10 {
            run();
        }
        let start = Instant::now();
        for _ in 0..iters {
            run();
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("{label:<26} {per:9.2} ns/call");
        per
    };
    let rows: [(&str, Box<dyn FnMut()>, Box<dyn FnMut()>); 6] = [
        (
            "stress  cauchy",
            Box::new(|| {
                black_box(ok(model.cauchy_stress(black_box(&f))));
            }),
            Box::new(|| {
                black_box(neo_hookean::cauchy_stress(&model, black_box(&f)));
            }),
        ),
        (
            "stress  first piola",
            Box::new(|| {
                black_box(ok(model.first_piola_kirchhoff_stress(black_box(&f))));
            }),
            Box::new(|| {
                black_box(neo_hookean::first_piola_kirchhoff_stress(
                    &model,
                    black_box(&f),
                ));
            }),
        ),
        (
            "stress  second piola",
            Box::new(|| {
                black_box(ok(model.second_piola_kirchhoff_stress(black_box(&f))));
            }),
            Box::new(|| {
                black_box(neo_hookean::second_piola_kirchhoff_stress(
                    &model,
                    black_box(&f),
                ));
            }),
        ),
        (
            "tangent cauchy",
            Box::new(|| {
                black_box(ok(model.cauchy_tangent_stiffness(black_box(&f))));
            }),
            Box::new(|| {
                black_box(neo_hookean::cauchy_tangent_stiffness(&model, black_box(&f)));
            }),
        ),
        (
            "tangent first piola",
            Box::new(|| {
                black_box(ok(
                    model.first_piola_kirchhoff_tangent_stiffness(black_box(&f))
                ));
            }),
            Box::new(|| {
                black_box(neo_hookean::first_piola_kirchhoff_tangent_stiffness(
                    &model,
                    black_box(&f),
                ));
            }),
        ),
        (
            "tangent second piola",
            Box::new(|| {
                black_box(ok(
                    model.second_piola_kirchhoff_tangent_stiffness(black_box(&f))
                ));
            }),
            Box::new(|| {
                black_box(neo_hookean::second_piola_kirchhoff_tangent_stiffness(
                    &model,
                    black_box(&f),
                ));
            }),
        ),
    ];
    for (label, hand, ad) in rows {
        let hand_ns = time(&format!("{label} hand"), hand);
        let ad_ns = time(&format!("{label} enzyme"), ad);
        println!("{label:<26} enzyme/hand = {:.2}x\n", ad_ns / hand_ns);
    }
}
