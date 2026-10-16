use super::{axpy, rank_one_quad, rank_two_quad};
use crate::math::Scalar;

// Lengths straddle every vector width and remainder loop.
const LENGTHS: [usize; 14] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 37];

fn values(seed: usize, length: usize) -> Vec<Scalar> {
    (0..length)
        .map(|i| ((i * 7 + seed * 13) % 11) as Scalar / 5.0 - 1.0)
        .collect()
}

fn close(a: &[Scalar], b: &[Scalar]) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())))
}

#[test]
fn axpy_matches_reference() {
    LENGTHS.iter().for_each(|&n| {
        let column = values(1, n);
        let mut target = values(2, n);
        let mut expected = target.clone();
        expected
            .iter_mut()
            .zip(&column)
            .for_each(|(t, c)| *t -= c * 0.75);
        axpy(&mut target, &column, 0.75);
        assert!(close(&target, &expected), "axpy, length {n}");
    })
}

#[test]
fn rank_one_quad_matches_reference() {
    let u = [0.5, -1.25, 2.0, 0.125];
    LENGTHS.iter().for_each(|&n| {
        let column = values(3, n);
        let mut temps: [Vec<Scalar>; 4] = std::array::from_fn(|k| values(4 + k, n));
        let mut expected = temps.clone();
        expected.iter_mut().zip(u).for_each(|(temp, u_k)| {
            temp.iter_mut()
                .zip(&column)
                .for_each(|(t, c)| *t += c * u_k)
        });
        let [t0, t1, t2, t3] = &mut temps;
        rank_one_quad(t0, t1, t2, t3, &column, u);
        temps
            .iter()
            .zip(&expected)
            .for_each(|(got, want)| assert!(close(got, want), "rank_one_quad, length {n}"));
    })
}

#[test]
fn rank_two_quad_matches_reference() {
    let u = [0.5, -1.25, 2.0, 0.125];
    let w = [-0.75, 1.5, 0.25, -2.0];
    LENGTHS.iter().for_each(|&n| {
        let column = values(3, n);
        let other = values(9, n);
        let mut temps: [Vec<Scalar>; 4] = std::array::from_fn(|k| values(4 + k, n));
        let mut expected = temps.clone();
        expected
            .iter_mut()
            .zip(u.iter().zip(&w))
            .for_each(|(temp, (u_k, w_k))| {
                temp.iter_mut()
                    .zip(column.iter().zip(&other))
                    .for_each(|(t, (c, o))| *t += c * u_k + o * w_k)
            });
        let [t0, t1, t2, t3] = &mut temps;
        rank_two_quad(t0, t1, t2, t3, &column, &other, u, w);
        temps
            .iter()
            .zip(&expected)
            .for_each(|(got, want)| assert!(close(got, want), "rank_two_quad, length {n}"));
    })
}
