use super::{collinear, exact, filtered, orient3d};
use crate::geometry::predicates::test::Lcg;
use core::{array::from_fn, cmp::Ordering};

fn floats(p: [i64; 3]) -> [f64; 3] {
    p.map(|x| x as f64)
}

fn reference(a: [i64; 3], b: [i64; 3], c: [i64; 3], d: [i64; 3]) -> Ordering {
    let [adx, ady, adz] = [0, 1, 2].map(|i| (a[i] - d[i]) as i128);
    let [bdx, bdy, bdz] = [0, 1, 2].map(|i| (b[i] - d[i]) as i128);
    let [cdx, cdy, cdz] = [0, 1, 2].map(|i| (c[i] - d[i]) as i128);
    let determinant = adz * (bdx * cdy - cdx * bdy)
        + bdz * (cdx * ady - adx * cdy)
        + cdz * (adx * bdy - bdx * ady);
    determinant.cmp(&0)
}

#[test]
fn a_point_below_the_plane_is_positive() {
    let (a, b, c) = (
        [0.0, 0.0, 0.0].into(),
        [1.0, 0.0, 0.0].into(),
        [0.0, 1.0, 0.0].into(),
    );
    assert_eq!(
        orient3d(&a, &b, &c, &[0.0, 0.0, -1.0].into()),
        Ordering::Greater
    );
    assert_eq!(
        orient3d(&a, &b, &c, &[0.0, 0.0, 1.0].into()),
        Ordering::Less
    );
    assert_eq!(
        orient3d(&a, &b, &c, &[0.3, 0.3, 0.0].into()),
        Ordering::Equal
    );
}

#[test]
fn swapping_two_points_reverses_the_sign_and_rotating_three_keeps_it() {
    let mut random = Lcg(1);
    (0..2000).for_each(|_| {
        let [a, b, c, d] = [0; 4].map(|_| [random.unit(), random.unit(), random.unit()].into());
        let sign = orient3d(&a, &b, &c, &d);
        assert_eq!(orient3d(&b, &a, &c, &d), sign.reverse());
        assert_eq!(orient3d(&b, &c, &a, &d), sign);
    })
}

#[test]
fn points_on_a_slanted_plane_are_exactly_coplanar() {
    let points = [
        [0.5, 0.25, 0.25],
        [0.125, 0.625, 0.25],
        [0.25, 0.25, 0.5],
        [0.5, 0.5, 0.0],
    ]
    .map(Into::into);
    assert_eq!(
        orient3d(&points[0], &points[1], &points[2], &points[3]),
        Ordering::Equal
    );
}

#[test]
fn the_filter_is_never_wrong() {
    let mut random = Lcg(2);
    (0..20000).for_each(|_| {
        let [a, b, c, d] = [0; 4].map(|_| [random.unit(), random.unit(), random.unit()]);
        if let Some(sign) = filtered(a, b, c, d) {
            assert_eq!(sign, exact(a, b, c, d));
        }
    })
}

#[test]
fn nearly_coplanar_integer_points_match_128_bit_arithmetic() {
    let mut random = Lcg(3);
    let (mut undecided, mut zero) = (0, 0);
    (0..20000).for_each(|_| {
        let [a, b, c] = [0; 3].map(|_| [0; 3].map(|_| random.integer(30)));
        let (s, t) = (random.integer(2), random.integer(2));
        let mut d: [i64; 3] = from_fn(|i| a[i] + s * (b[i] - a[i]) + t * (c[i] - a[i]));
        d[(random.bits() % 3) as usize] += random.integer(1);
        let expected = reference(a, b, c, d);
        let [fa, fb, fc, fd] = [a, b, c, d].map(floats);
        assert_eq!(exact(fa, fb, fc, fd), expected);
        assert_eq!(
            orient3d(&fa.into(), &fb.into(), &fc.into(), &fd.into()),
            expected
        );
        undecided += filtered(fa, fb, fc, fd).is_none() as usize;
        zero += (expected == Ordering::Equal) as usize;
    });
    assert!(undecided > 1000);
    assert!(zero > 1000);
}

#[test]
fn large_integer_points_match_128_bit_arithmetic() {
    let mut random = Lcg(4);
    (0..5000).for_each(|_| {
        let [a, b, c, d] = [0; 4].map(|_| [0; 3].map(|_| random.integer(37)));
        let expected = reference(a, b, c, d);
        let [a, b, c, d] = [a, b, c, d].map(floats);
        assert_eq!(exact(a, b, c, d), expected);
    })
}

#[test]
fn points_on_a_line_are_collinear_and_a_nudge_is_not() {
    let (a, b) = ([0.0, 0.0, 0.0].into(), [1.0, 2.0, 3.0].into());
    assert!(collinear(&a, &b, &[3.0, 6.0, 9.0].into()));
    assert!(!collinear(&a, &b, &[3.0, 6.0, 9.0 + 2e-15].into()));
    assert!(collinear(&a, &a, &b));
}

#[test]
fn collinearity_of_large_integer_points_matches_128_bit_arithmetic() {
    let mut random = Lcg(6);
    let (mut lines, mut nudged) = (0, 0);
    (0..5000).for_each(|_| {
        let a = [0; 3].map(|_| random.integer(30));
        let direction = [0; 3].map(|_| random.integer(8));
        let (s, t) = (random.integer(3), random.integer(3));
        let b: [i64; 3] = from_fn(|i| a[i] + s * direction[i]);
        let mut c: [i64; 3] = from_fn(|i| a[i] + t * direction[i]);
        c[(random.bits() % 3) as usize] += random.integer(1);
        let [u, v]: [[i128; 3]; 2] = [b, c].map(|p| from_fn(|i| (p[i] - a[i]) as i128));
        let expected =
            u[1] * v[2] == u[2] * v[1] && u[2] * v[0] == u[0] * v[2] && u[0] * v[1] == u[1] * v[0];
        lines += expected as usize;
        nudged += !expected as usize;
        let [a, b, c] = [a, b, c].map(floats);
        assert_eq!(collinear(&a.into(), &b.into(), &c.into()), expected);
    });
    assert!(lines > 500 && nudged > 500);
}
