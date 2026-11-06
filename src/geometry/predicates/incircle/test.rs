use super::{exact, filtered, incircle, incircle_metric};
use crate::geometry::{Coordinate, predicates::test::Lcg};
use core::{array::from_fn, cmp::Ordering, f64::consts::TAU};

fn circle() -> Vec<[i64; 2]> {
    (-65..=65)
        .flat_map(|x| (-65..=65).map(move |y| [x, y]))
        .filter(|[x, y]| x * x + y * y == 65 * 65)
        .collect()
}

fn reference(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2], gram: [i64; 3]) -> Ordering {
    let [adx, ady] = from_fn(|i| (a[i] - d[i]) as i128);
    let [bdx, bdy] = from_fn(|i| (b[i] - d[i]) as i128);
    let [cdx, cdy] = from_fn(|i| (c[i] - d[i]) as i128);
    let [g0, g1, g2] = gram.map(|g| g as i128);
    let lift = |x: i128, y: i128| g0 * x * x + 2 * g1 * x * y + g2 * y * y;
    let determinant = lift(adx, ady) * (bdx * cdy - cdx * bdy)
        + lift(bdx, bdy) * (cdx * ady - adx * cdy)
        + lift(cdx, cdy) * (adx * bdy - bdx * ady);
    determinant.cmp(&0)
}

fn floats(p: [i64; 2]) -> [f64; 2] {
    p.map(|x| x as f64)
}

fn lifted(p: [f64; 2]) -> Coordinate<3> {
    [p[0], p[1], 5.0].into()
}

#[test]
fn a_point_inside_the_circle_is_positive_and_on_it_is_zero() {
    let (a, b, c) = (lifted([1.0, 0.0]), lifted([0.0, 1.0]), lifted([-1.0, 0.0]));
    assert_eq!(
        incircle(&a, &b, &c, &lifted([0.0, 0.0]), 2),
        Ordering::Greater
    );
    assert_eq!(incircle(&a, &b, &c, &lifted([2.0, 0.0]), 2), Ordering::Less);
    assert_eq!(
        incircle(&a, &b, &c, &lifted([0.0, -1.0]), 2),
        Ordering::Equal
    );
}

#[test]
fn swapping_two_points_reverses_the_sign_and_rotating_three_keeps_it() {
    let mut random = Lcg(1);
    (0..2000).for_each(|_| {
        let [a, b, c, d] = [0; 4].map(|_| lifted([random.unit(), random.unit()]));
        let sign = incircle(&a, &b, &c, &d, 2);
        assert_eq!(incircle(&b, &a, &c, &d, 2), sign.reverse());
        assert_eq!(incircle(&b, &c, &a, &d, 2), sign);
    })
}

#[test]
fn dropping_a_different_axis_reads_the_remaining_coordinates_in_cyclic_order() {
    let mut random = Lcg(2);
    (0..2000).for_each(|_| {
        let [a, b, c, d] = [0; 4].map(|_| [random.unit(), random.unit(), random.unit()]);
        let rotated = |p: [f64; 3]| [p[1], p[2], p[0]];
        assert_eq!(
            incircle(&a.into(), &b.into(), &c.into(), &d.into(), 0),
            incircle(
                &rotated(a).into(),
                &rotated(b).into(),
                &rotated(c).into(),
                &rotated(d).into(),
                2
            )
        );
    })
}

#[test]
fn a_stretched_metric_changes_which_points_are_on_the_circle() {
    let (a, b, c) = ([1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]);
    let stretched = [4.0, 0.0, 1.0];
    assert_eq!(
        incircle_metric(a, b, c, [0.0, 0.0], stretched),
        Ordering::Greater
    );
    assert_eq!(
        incircle_metric(a, b, c, [0.0, -2.0], stretched),
        Ordering::Equal
    );
    assert_ne!(
        incircle_metric(a, b, c, [0.0, -2.0], [1.0, 0.0, 1.0]),
        Ordering::Equal
    );
}

#[test]
fn cocircular_lattice_points_and_their_nudges_match_128_bit_arithmetic() {
    let mut random = Lcg(3);
    let circle = circle();
    let (mut undecided, mut zero, mut nonzero) = (0, 0, 0);
    (0..20000).for_each(|_| {
        let scale = random.integer(2).abs() + 1;
        let shift = [random.integer(18), random.integer(18)];
        let pick = |random: &mut Lcg| {
            let point = circle[(random.bits() as usize) % circle.len()];
            [point[0] * scale + shift[0], point[1] * scale + shift[1]]
        };
        let [a, b, c] = [0; 3].map(|_| pick(&mut random));
        let mut d = pick(&mut random);
        d[(random.bits() % 2) as usize] += random.integer(1);
        let expected = reference(a, b, c, d, [1, 0, 1]);
        let [fa, fb, fc, fd] = [a, b, c, d].map(floats);
        assert_eq!(exact(fa, fb, fc, fd, [1.0, 0.0, 1.0]), expected);
        assert_eq!(incircle_metric(fa, fb, fc, fd, [1.0, 0.0, 1.0]), expected);
        assert_eq!(
            incircle(&lifted(fa), &lifted(fb), &lifted(fc), &lifted(fd), 2),
            expected
        );
        undecided += filtered(fa, fb, fc, fd, [1.0, 0.0, 1.0]).is_none() as usize;
        zero += (expected == Ordering::Equal) as usize;
        nonzero += (expected != Ordering::Equal) as usize;
    });
    assert!(undecided > 1000 && zero > 1000 && nonzero > 1000);
}

#[test]
fn a_sheared_metric_keeps_the_images_of_cocircular_points_cocircular() {
    let mut random = Lcg(4);
    let circle = circle();
    let (mut zero, mut nonzero) = (0, 0);
    (0..20000).for_each(|_| {
        let shear = random.integer(2).abs() + 1;
        let gram = [1, shear, 1 + shear * shear];
        let pick = |random: &mut Lcg| {
            let [p, q] = circle[(random.bits() as usize) % circle.len()];
            [p - shear * q, q]
        };
        let [a, b, c] = [0; 3].map(|_| pick(&mut random));
        let mut d = pick(&mut random);
        d[(random.bits() % 2) as usize] += random.integer(1);
        let expected = reference(a, b, c, d, gram);
        let [fa, fb, fc, fd] = [a, b, c, d].map(floats);
        let fgram = gram.map(|g| g as f64);
        assert_eq!(exact(fa, fb, fc, fd, fgram), expected);
        assert_eq!(incircle_metric(fa, fb, fc, fd, fgram), expected);
        zero += (expected == Ordering::Equal) as usize;
        nonzero += (expected != Ordering::Equal) as usize;
    });
    assert!(zero > 1000 && nonzero > 1000);
}

fn near_circle(
    random: &mut Lcg,
    centre: [f64; 2],
    radius: f64,
    map: impl Fn([f64; 2]) -> [f64; 2],
) -> [f64; 2] {
    let angle = TAU * random.unit();
    let [x, y] = map([radius * angle.cos(), radius * angle.sin()]);
    [centre[0] + x, centre[1] + y]
}

#[test]
fn the_filter_is_never_wrong_on_points_cocircular_up_to_rounding() {
    let mut random = Lcg(5);
    let (mut decided, mut undecided) = (0, 0);
    (0..100000).for_each(|_| {
        let centre = [random.unit(), random.unit()];
        let radius = 0.1 + 100.0 * random.unit();
        let [a, b, c, d] = [0; 4].map(|_| near_circle(&mut random, centre, radius, |p| p));
        match filtered(a, b, c, d, [1.0, 0.0, 1.0]) {
            Some(sign) => {
                decided += 1;
                assert_eq!(sign, exact(a, b, c, d, [1.0, 0.0, 1.0]));
            }
            None => undecided += 1,
        }
    });
    assert!(decided > 1000 && undecided > 1000);
}

#[test]
fn the_filter_is_never_wrong_in_a_stretched_metric() {
    let mut random = Lcg(6);
    let (mut decided, mut undecided) = (0, 0);
    (0..100000).for_each(|_| {
        let (m00, m01, m11) = (
            0.2 + 3.0 * random.unit(),
            -2.0 + 4.0 * random.unit(),
            0.2 + 3.0 * random.unit(),
        );
        let gram = [m00 * m00, m00 * m01, m01 * m01 + m11 * m11];
        let inverse = |[x, y]: [f64; 2]| [x / m00 - m01 * y / (m00 * m11), y / m11];
        let centre = [random.unit(), random.unit()];
        let radius = 0.1 + 10.0 * random.unit();
        let [a, b, c, d] = [0; 4].map(|_| near_circle(&mut random, centre, radius, inverse));
        match filtered(a, b, c, d, gram) {
            Some(sign) => {
                decided += 1;
                assert_eq!(sign, exact(a, b, c, d, gram));
            }
            None => undecided += 1,
        }
    });
    assert!(decided > 1000 && undecided > 1000);
}
