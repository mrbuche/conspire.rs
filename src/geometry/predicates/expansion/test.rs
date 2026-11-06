use super::Expansion;
use core::cmp::Ordering;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn wild(&mut self) -> f64 {
        let sign = if self.next() < 0.5 { -1.0 } else { 1.0 };
        sign * self.next() * 2f64.powi((self.next() * 80.0) as i32 - 40)
    }
}

#[test]
fn a_sum_keeps_what_floating_point_would_lose() {
    let sum = Expansion::new(1e16).sum(Expansion::new(1.0));
    assert_eq!(sum.minus(Expansion::new(1e16)).sign(), Ordering::Greater);
    assert_eq!(1e16 + 1.0 - 1e16, 0.0);
}

#[test]
fn sums_and_differences_cancel_exactly() {
    let mut random = Lcg(3);
    (0..2000).for_each(|_| {
        let (a, b, c) = (random.wild(), random.wild(), random.wild());
        let total = Expansion::new(a)
            .sum(Expansion::new(b))
            .sum(Expansion::new(c));
        let back = total
            .minus(Expansion::new(c))
            .minus(Expansion::new(a))
            .minus(Expansion::new(b));
        assert_eq!(back.sign(), Ordering::Equal);
    })
}

#[test]
fn a_product_keeps_its_rounding_error() {
    let a = 1.0 + 2f64.powi(-30);
    let square = Expansion::product(a, a);
    let rounded = square.minus(Expansion::new(1.0 + 2f64.powi(-29)));
    assert_eq!(rounded.sign(), Ordering::Greater);
    assert_eq!(
        rounded.minus(Expansion::new(2f64.powi(-60))).sign(),
        Ordering::Equal
    );
}

#[test]
fn a_difference_of_squares_factors_exactly() {
    let mut random = Lcg(11);
    (0..2000).for_each(|_| {
        let (x, y) = (random.wild(), random.wild());
        let (x, y) = (Expansion::new(x), Expansion::new(y));
        let factored = x.clone().sum(y.clone()).times(&x.clone().minus(y.clone()));
        let expanded = x.times(&x).minus(y.times(&y));
        assert_eq!(factored.minus(expanded).sign(), Ordering::Equal);
    })
}

#[test]
fn the_sign_follows_the_number() {
    let mut random = Lcg(5);
    (0..2000).for_each(|_| {
        let (a, b) = (random.wild(), random.wild());
        let difference = Expansion::difference(a, b);
        let expected = a.partial_cmp(&b).unwrap();
        assert_eq!(difference.sign(), expected);
        assert_eq!(difference.negated().sign(), expected.reverse());
    })
}
