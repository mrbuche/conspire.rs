use super::super::pierce::Local;
use super::clip;
use crate::geometry::Coordinate;
use core::array::from_fn;

struct Lcg(u64);

impl Lcg {
    fn integer(&mut self, bound: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % (2 * bound as u64 + 1)) as i64 - bound
    }
    fn point(&mut self, bound: i64) -> [i64; 2] {
        [self.integer(bound), self.integer(bound)]
    }
}

type Fraction = (i128, i128);

fn less(a: Fraction, b: Fraction) -> bool {
    a.0 * b.1 < b.0 * a.1
}

fn orient(a: [i128; 2], b: [i128; 2], c: [i128; 2]) -> i128 {
    (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
}

fn wide(point: [i64; 2]) -> [i128; 2] {
    point.map(|x| x as i128)
}

fn reference(
    triangle: [[i64; 2]; 3],
    start: [i64; 2],
    finish: [i64; 2],
) -> Option<(Fraction, Fraction)> {
    let [t0, t1, t2] = triangle.map(wide);
    let (start, finish) = (wide(start), wide(finish));
    let turn = orient(t0, t1, t2).signum();
    let (mut low, mut high): (Fraction, Fraction) = ((0, 1), (1, 1));
    for edge in 0..3 {
        let (a, b) = ([t0, t1, t2][edge], [t0, t1, t2][(edge + 1) % 3]);
        let at_start = orient(a, b, start) * turn;
        let change = orient(a, b, finish) * turn - at_start;
        if change == 0 {
            if at_start < 0 {
                return None;
            }
        } else if change > 0 {
            let bound = (-at_start, change);
            if less(low, bound) {
                low = bound;
            }
        } else {
            let bound = (at_start, -change);
            if less(bound, high) {
                high = bound;
            }
        }
    }
    less(low, high).then_some((low, high))
}

fn classify(
    triangle: [[i64; 2]; 3],
    start: [i64; 2],
    finish: [i64; 2],
    (numerator, denominator): Fraction,
) -> (Local, Local) {
    let [t0, t1, t2] = triangle.map(wide);
    let (start, finish) = (wide(start), wide(finish));
    let turn = orient(t0, t1, t2).signum();
    let scaled = |p: [i128; 2]| p.map(|x| x * denominator);
    let point: [i128; 2] = from_fn(|i| start[i] * denominator + numerator * (finish[i] - start[i]));
    let corners = [t0, t1, t2].map(scaled);
    let zeros: Vec<usize> = (0..3)
        .filter(|&edge| orient(corners[edge], corners[(edge + 1) % 3], point) * turn == 0)
        .collect();
    let on_triangle = match zeros[..] {
        [] => Local::Face,
        [edge] => Local::Edge(edge),
        _ => Local::Vertex(
            (0..3)
                .find(|&edge| zeros.contains(&edge) && zeros.contains(&((edge + 1) % 3)))
                .map(|edge| (edge + 1) % 3)
                .unwrap(),
        ),
    };
    let on_segment = if numerator == 0 {
        Local::Vertex(0)
    } else if numerator == denominator {
        Local::Vertex(1)
    } else {
        Local::Edge(0)
    };
    (on_triangle, on_segment)
}

fn check(embed: impl Fn([i64; 2]) -> [f64; 3], axis: usize, seed: u64) -> [usize; 4] {
    let mut random = Lcg(seed);
    let (mut clipped, mut at_vertices, mut on_edges, mut along_the_segment) = (0, 0, 0, 0);
    (0..30000).for_each(|_| {
        let triangle = [random.point(3), random.point(3), random.point(3)];
        let (start, finish) = (random.point(4), random.point(4));
        let [t0, t1, t2] = triangle.map(wide);
        if orient(t0, t1, t2) == 0 || start == finish {
            return;
        }
        let corners = triangle.map(|p| Coordinate::from(embed(p)));
        let (from, to) = (
            Coordinate::from(embed(start)),
            Coordinate::from(embed(finish)),
        );
        let found = clip(&from, &to, [&corners[0], &corners[1], &corners[2]], axis);
        let expected = reference(triangle, start, finish);
        match (expected, found) {
            (None, None) => (),
            (Some((low, high)), Some(ends)) => {
                clipped += 1;
                [(low, &ends[0]), (high, &ends[1])]
                    .iter()
                    .for_each(|&(at, end)| {
                        let (on_triangle, on_segment) = classify(triangle, start, finish, at);
                        assert_eq!(
                            (end.triangle, end.segment),
                            (on_triangle, on_segment),
                            "{triangle:?} {start:?} {finish:?}"
                        );
                        at_vertices += matches!(on_triangle, Local::Vertex(_)) as usize;
                        on_edges += matches!(on_triangle, Local::Edge(_)) as usize;
                        along_the_segment += matches!(on_segment, Local::Edge(_)) as usize;
                        let t = at.0 as f64 / at.1 as f64;
                        let (a, b) = (embed(start), embed(finish));
                        (0..3).for_each(|i| {
                            assert!(
                                (end.point[i].value() - (a[i] + t * (b[i] - a[i]))).abs() < 1e-9
                            )
                        });
                    })
            }
            (expected, found) => panic!(
                "{triangle:?} {start:?} {finish:?}: expected {expected:?}, found {}",
                found.is_some()
            ),
        }
    });
    [clipped, at_vertices, on_edges, along_the_segment]
}

#[test]
fn clipping_matches_exact_fractions_in_the_horizontal_plane() {
    let [clipped, vertices, edges, along] = check(|[u, v]| [u as f64, v as f64, 0.0], 2, 1);
    assert!(clipped > 3000 && vertices > 500 && edges > 500 && along > 500);
}

#[test]
fn clipping_matches_exact_fractions_in_a_tilted_plane_dropping_z() {
    check(|[u, v]| [u as f64, v as f64, (u + 2 * v) as f64], 2, 2);
}

#[test]
fn clipping_matches_exact_fractions_dropping_x() {
    check(|[u, v]| [(3 * u + v) as f64, u as f64, v as f64], 0, 3);
}

#[test]
fn clipping_matches_exact_fractions_dropping_y() {
    check(|[u, v]| [v as f64, (3 * u + v) as f64, u as f64], 1, 4);
}
