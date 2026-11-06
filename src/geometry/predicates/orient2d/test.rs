use super::{estimate, exact, filtered, orient2d};
use crate::geometry::predicates::test::Lcg;
use core::{array::from_fn, cmp::Ordering};

#[test]
fn counterclockwise_is_positive_in_the_plane_left_after_dropping_an_axis() {
    let (a, b, c) = (
        [0.0, 0.0, 7.0].into(),
        [1.0, 0.0, 7.0].into(),
        [0.0, 1.0, 7.0].into(),
    );
    assert_eq!(orient2d(&a, &b, &c, 2), Ordering::Greater);
    assert_eq!(orient2d(&b, &a, &c, 2), Ordering::Less);
    assert_eq!(
        orient2d(&a, &b, &[2.0, 0.0, -3.0].into(), 2),
        Ordering::Equal
    );
}

#[test]
fn dropping_a_different_axis_reads_the_remaining_coordinates_in_cyclic_order() {
    let mut random = Lcg(7);
    (0..2000).for_each(|_| {
        let [a, b, c] = [0; 3].map(|_| [random.unit(), random.unit(), random.unit()]);
        let rotated = |p: [f64; 3]| [p[1], p[2], p[0]];
        assert_eq!(
            orient2d(&a.into(), &b.into(), &c.into(), 0),
            orient2d(
                &rotated(a).into(),
                &rotated(b).into(),
                &rotated(c).into(),
                2
            )
        );
    })
}

#[test]
fn the_filter_is_never_wrong_on_points_collinear_up_to_rounding() {
    let mut random = Lcg(10);
    let (mut decided, mut undecided) = (0, 0);
    (0..100000).for_each(|_| {
        let a = [random.unit(), random.unit()];
        let scale = 1.0 + 1000.0 * random.unit();
        let b = [a[0] + scale * random.unit(), a[1] + scale * random.unit()];
        let t = random.unit();
        let c = from_fn(|i| a[i] + (b[i] - a[i]) * t);
        match filtered(a, b, c) {
            Some(sign) => {
                decided += 1;
                assert_eq!(sign, exact(a, b, c));
            }
            None => undecided += 1,
        }
    });
    assert!(decided > 1000 && undecided > 1000);
}

#[test]
fn consecutive_fibonacci_vectors_have_a_determinant_of_one_that_floating_point_loses() {
    let mut random = Lcg(9);
    let mut fibonacci = vec![0i64, 1];
    (0..76).for_each(|_| {
        fibonacci.push(fibonacci[fibonacci.len() - 1] + fibonacci[fibonacci.len() - 2])
    });
    let mut wrong = 0;
    (30..=72).for_each(|n| {
        let shift = [random.integer(40), random.integer(40)];
        let [c, a, b] = [
            [0, 0],
            [fibonacci[n + 1], fibonacci[n]],
            [fibonacci[n + 2], fibonacci[n + 1]],
        ]
        .map(|p| [(p[0] + shift[0]) as f64, (p[1] + shift[1]) as f64]);
        let expected = if n % 2 == 0 {
            Ordering::Greater
        } else {
            Ordering::Less
        };
        assert_eq!(exact(a, b, c), expected, "n = {n}");
        let lifted = |p: [f64; 2]| [p[0], p[1], 0.0].into();
        assert_eq!(
            orient2d(&lifted(a), &lifted(b), &lifted(c), 2),
            expected,
            "n = {n}"
        );
        let (determinant, _) = estimate(a, b, c);
        wrong += (determinant.partial_cmp(&0.0) != Some(expected)) as usize;
    });
    assert!(wrong > 5);
}

#[test]
fn nearly_collinear_large_integer_points_match_128_bit_arithmetic() {
    let mut random = Lcg(8);
    let (mut undecided, mut zero) = (0, 0);
    (0..20000).for_each(|_| {
        let [a, b] = [0; 2].map(|_| [0; 2].map(|_| random.integer(49)));
        let s = random.integer(1);
        let mut c: [i64; 2] = from_fn(|i| a[i] + s * (b[i] - a[i]));
        c[(random.bits() % 2) as usize] += random.integer(1);
        let determinant = (a[0] - c[0]) as i128 * (b[1] - c[1]) as i128
            - (a[1] - c[1]) as i128 * (b[0] - c[0]) as i128;
        let [fa, fb, fc] = [a, b, c].map(|p| p.map(|x| x as f64));
        assert_eq!(exact(fa, fb, fc), determinant.cmp(&0));
        let lifted = |p: [f64; 2]| [p[0], p[1], 0.0].into();
        assert_eq!(
            orient2d(&lifted(fa), &lifted(fb), &lifted(fc), 2),
            determinant.cmp(&0)
        );
        undecided += filtered(fa, fb, fc).is_none() as usize;
        zero += (determinant == 0) as usize;
    });
    assert!(undecided > 1000 && zero > 1000);
}
