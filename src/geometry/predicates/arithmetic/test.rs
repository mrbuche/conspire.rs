use super::{Arithmetic, Estimate};
use crate::geometry::predicates::{HALF_EPSILON, expansion::Expansion, test::Lcg};
use core::cmp::Ordering;

/// A float of any size from about 2⁻¹⁶ to 2¹⁶, of either sign.
fn float(random: &mut Lcg) -> f64 {
    (random.unit() - 0.5) * 2.0_f64.powi(random.integer(4) as i32)
}

/// A random formula of differences, sums and products, in any arithmetic. Built twice from the
/// same seed it is the same formula.
fn formula<A: Arithmetic>(random: &mut Lcg, depth: usize) -> A {
    if depth == 0 || random.bits().is_multiple_of(5) {
        return if random.bits().is_multiple_of(2) {
            A::difference(float(random), float(random))
        } else {
            A::exactly(float(random))
        };
    }
    let (left, right) = (
        formula::<A>(random, depth - 1),
        formula::<A>(random, depth - 1),
    );
    match random.bits() % 4 {
        0 => left.sum(right),
        1 => left.minus(right),
        _ => left.times(&right),
    }
}

#[test]
fn an_estimate_is_within_its_error_of_the_exact_value() {
    for seed in 0..4000 {
        let estimate: Estimate = formula(&mut Lcg(seed), 5);
        let exact: Expansion = formula(&mut Lcg(seed), 5);
        let expected = exact.estimate();
        // The exact value is itself only rounded once, to within half a unit in the last place.
        let slack = HALF_EPSILON * expected.abs();
        assert!(
            (estimate.value() - expected).abs() <= estimate.error() + slack,
            "{seed}: {} against {expected}, error {}",
            estimate.value(),
            estimate.error()
        );
    }
}

#[test]
fn a_sign_that_is_certain_is_the_exact_sign() {
    let mut certain = 0;
    for seed in 0..20000 {
        let estimate: Estimate = formula(&mut Lcg(seed), 4);
        let exact: Expansion = formula(&mut Lcg(seed), 4);
        if let Some(sign) = estimate.certain_sign() {
            certain += 1;
            assert_eq!(sign, exact.sign(), "{seed}");
        }
    }
    assert!(certain > 10000, "{certain}");
}

/// `(a + b) c - (a c + b c)`, which is zero, or `ulps` units in the last place from it.
fn nearly_distributive<A: Arithmetic>(a: f64, b: f64, c: f64, ulps: f64) -> A {
    let (a, b, c) = (A::exactly(a), A::exactly(b), A::exactly(c));
    let right = a
        .times(&c)
        .sum(b.times(&c))
        .sum(A::exactly(ulps * f64::EPSILON));
    a.sum(b).times(&c).minus(right)
}

#[test]
fn a_sum_that_cancels_is_never_certain_of_a_sign_it_does_not_have() {
    let mut random = Lcg(7);
    let (mut zeros, mut undecided) = (0, 0);
    for case in 0..20000 {
        let (a, b, c) = (float(&mut random), float(&mut random), float(&mut random));
        let ulps = [0.0, 0.0, 1.0, -1.0, 4.0, -4.0][case % 6];
        let estimate: Estimate = nearly_distributive(a, b, c, ulps);
        let exact: Expansion = nearly_distributive(a, b, c, ulps);
        zeros += (exact.sign() == Ordering::Equal) as usize;
        match estimate.certain_sign() {
            Some(sign) => assert_eq!(sign, exact.sign(), "{a} {b} {c} {ulps}"),
            None => undecided += 1,
        }
    }
    // Both are common: exact zeros, which an estimate can never certify, and certain signs.
    assert!(zeros > 1000 && undecided > 1000, "{zeros} {undecided}");
}

#[test]
fn compressing_keeps_the_number_and_does_not_lengthen_it() {
    let mut random = Lcg(11);
    let mut shortened = 0;
    for _ in 0..2000 {
        let (a, b, c, d) = (
            float(&mut random),
            float(&mut random),
            float(&mut random),
            float(&mut random),
        );
        let x = Expansion::difference(a, b).times(&Expansion::difference(c, d));
        let y = x
            .times(&Expansion::difference(b, c))
            .sum(Expansion::difference(a, d).times(&x));
        for expansion in [x, y] {
            let compressed = expansion.clone().compressed();
            assert_eq!(
                expansion.clone().minus(compressed.clone()).sign(),
                Ordering::Equal
            );
            assert!(compressed.len() <= expansion.len());
            shortened += (compressed.len() < expansion.len()) as usize;
        }
    }
    assert!(shortened > 200, "{shortened}");
}

#[test]
fn compressing_nothing_or_one_component_changes_nothing() {
    assert_eq!(Expansion::new(0.0).compressed().len(), 0);
    assert_eq!(Expansion::new(3.5).compressed().estimate(), 3.5);
}

#[test]
fn compressing_an_expansion_that_cancels_gives_zero() {
    let x = Expansion::difference(1.0, 1e-30).times(&Expansion::difference(1.0, 1e-30));
    let y = Expansion::difference(1.0, 1e-30).times(&Expansion::difference(1.0, 1e-30));
    let zero = x.minus(y).compressed();
    assert_eq!(zero.sign(), Ordering::Equal);
    assert_eq!(zero.len(), 0);
}
