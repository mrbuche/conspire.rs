use super::{Cutter, along, decide};
use crate::geometry::{
    Coordinate,
    predicates::{arithmetic::Estimate, expansion::Expansion, test::Lcg},
};
use core::{array::from_fn, cmp::Ordering};

type Vector = [i64; 3];

fn point(v: Vector) -> Coordinate<3> {
    v.map(|x| x as f64).into()
}

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vector, b: Vector) -> i64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn plus(a: Vector, scale: i64, b: Vector) -> Vector {
    from_fn(|i| a[i] + scale * b[i])
}

enum Owned {
    Face([Coordinate<3>; 3]),
    Edge([Coordinate<3>; 2]),
    Vertex(Coordinate<3>),
}

impl Owned {
    fn cutter(&self) -> Cutter<'_> {
        match self {
            Owned::Face([a, b, c]) => Cutter::Face([a, b, c]),
            Owned::Edge([r, s]) => Cutter::Edge([r, s]),
            Owned::Vertex(v) => Cutter::Vertex(v),
        }
    }
}

fn vector(random: &mut Lcg) -> Vector {
    loop {
        let v = from_fn(|_| random.integer(3));
        if v != [0; 3] {
            return v;
        }
    }
}

fn through(random: &mut Lcg, kind: usize, x: Vector, line: Vector) -> Owned {
    match kind {
        0 => loop {
            let (e1, e2) = (vector(random), vector(random));
            if dot(cross(e1, e2), line) != 0 {
                break Owned::Face([
                    point(plus(x, 1, e1)),
                    point(plus(x, 1, e2)),
                    point(plus(plus(x, -1, e1), -1, e2)),
                ]);
            }
        },
        1 => loop {
            let d = vector(random);
            if cross(d, line) != [0; 3] {
                let (before, after) = (random.integer(2).abs() + 1, random.integer(2).abs() + 1);
                break Owned::Edge([point(plus(x, -before, d)), point(plus(x, after, d))]);
            }
        },
        _ => Owned::Vertex(point(x)),
    }
}

struct Edge {
    origin: Vector,
    line: Vector,
    length: i64,
}

impl Edge {
    fn new(random: &mut Lcg) -> Self {
        Self {
            origin: from_fn(|_| random.integer(28)),
            line: vector(random),
            length: 1 << (10 + random.bits() % 11),
        }
    }
    fn ends(&self) -> [Coordinate<3>; 2] {
        [
            point(self.origin),
            point(plus(self.origin, self.length, self.line)),
        ]
    }
    fn at(&self, count: i64) -> Vector {
        plus(self.origin, count, self.line)
    }
}

fn check(edge: &Edge, random: &mut Lcg, i: i64, j: i64) {
    let [p, q] = edge.ends();
    let (kind_x, kind_y) = ((random.bits() % 3) as usize, (random.bits() % 3) as usize);
    let x = through(random, kind_x, edge.at(i), edge.line);
    let y = through(random, kind_y, edge.at(j), edge.line);
    let expected = i.cmp(&j);
    let found = along(&p, &q, &x.cutter(), &y.cutter()).unwrap();
    assert_eq!(
        found, expected,
        "{kind_x} {kind_y} at {i} and {j} of {}",
        edge.length
    );
    assert_eq!(
        along(&p, &q, &y.cutter(), &x.cutter()).unwrap(),
        expected.reverse()
    );
    assert_eq!(
        along(&q, &p, &x.cutter(), &y.cutter()).unwrap(),
        expected.reverse()
    );
}

#[test]
fn points_along_an_edge_are_ordered_whatever_cuts_them() {
    let mut random = Lcg(1);
    (0..4000).for_each(|_| {
        let edge = Edge::new(&mut random);
        let span = edge.length - 1;
        let i = 1 + (random.bits() as i64).rem_euclid(span);
        let j = 1 + (random.bits() as i64).rem_euclid(span);
        check(&edge, &mut random, i, j);
    })
}

#[test]
fn neighbouring_points_on_a_long_edge_are_ordered() {
    let mut random = Lcg(2);
    (0..4000).for_each(|_| {
        let mut edge = Edge::new(&mut random);
        edge.length = 1 << 20;
        let i = 1 + (random.bits() as i64).rem_euclid(edge.length - 2);
        check(&edge, &mut random, i, i + 1);
    })
}

#[test]
fn the_same_point_cut_by_different_things_is_equal() {
    let mut random = Lcg(3);
    (0..2000).for_each(|_| {
        let edge = Edge::new(&mut random);
        let i = 1 + (random.bits() as i64).rem_euclid(edge.length - 1);
        check(&edge, &mut random, i, i);
    })
}

#[test]
fn an_edge_along_an_axis_is_ordered_by_a_vertex() {
    let mut random = Lcg(4);
    [[0, 3, -5], [0, 0, 7], [2, 0, 0], [0, -4, 0]]
        .iter()
        .for_each(|&line| {
            (0..300).for_each(|_| {
                let mut edge = Edge::new(&mut random);
                edge.line = line;
                let span = edge.length - 1;
                let i = 1 + (random.bits() as i64).rem_euclid(span);
                let j = 1 + (random.bits() as i64).rem_euclid(span);
                check(&edge, &mut random, i, j);
            })
        })
}

#[test]
fn a_cutter_the_line_is_parallel_to_is_an_error() {
    let mut random = Lcg(5);
    let edge = Edge::new(&mut random);
    let [p, q] = edge.ends();
    let good = through(&mut random, 2, edge.at(3), edge.line);
    let off = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        .into_iter()
        .find(|&axis| cross(edge.line, axis) != [0; 3])
        .unwrap();
    let parallel_plane = Owned::Face([
        point(plus(edge.origin, 1, off)),
        point(plus(plus(edge.origin, 1, off), 1, edge.line)),
        point(plus(plus(edge.origin, 2, off), 1, edge.line)),
    ]);
    let along_the_line = Owned::Edge([point(edge.origin), point(plus(edge.origin, 2, edge.line))]);
    [parallel_plane, along_the_line].iter().for_each(|bad| {
        let fails = |x: &Owned, y: &Owned| along(&p, &q, &x.cutter(), &y.cutter()).is_err();
        assert!(fails(bad, &good));
        assert!(fails(&good, bad));
    });
    assert!(along(&p, &p, &good.cutter(), &good.cutter()).is_err());
}

impl Owned {
    fn nudged(&self, random: &mut Lcg) -> Owned {
        let mut nudge = |point: &Coordinate<3>| -> Coordinate<3> {
            from_fn(|axis| {
                let x = point[axis].value();
                let ulps = random.integer(2) as f64;
                x + ulps * x.abs().max(1.0) * f64::EPSILON
            })
            .into()
        };
        match self {
            Owned::Face([a, b, c]) => Owned::Face([nudge(a), nudge(b), nudge(c)]),
            Owned::Edge([r, s]) => Owned::Edge([nudge(r), nudge(s)]),
            Owned::Vertex(v) => Owned::Vertex(nudge(v)),
        }
    }
}

#[test]
fn the_filter_is_never_certain_of_a_wrong_order() {
    let mut random = Lcg(6);
    let (mut decided, mut cases) = (0, 0);
    for _ in 0..20000 {
        let edge = Edge::new(&mut random);
        let span = edge.length - 1;
        let i = 1 + (random.bits() as i64).rem_euclid(span);
        let j = if random.bits().is_multiple_of(2) {
            i
        } else {
            1 + (random.bits() as i64).rem_euclid(span)
        };
        let [p, q] = edge.ends();
        let (kind_x, kind_y) = ((random.bits() % 3) as usize, (random.bits() % 3) as usize);
        let x = through(&mut random, kind_x, edge.at(i), edge.line).nudged(&mut random);
        let y = through(&mut random, kind_y, edge.at(j), edge.line).nudged(&mut random);
        let filtered = decide::<Estimate>(&p, &q, &x.cutter(), &y.cutter());
        let exact = decide::<Expansion>(&p, &q, &x.cutter(), &y.cutter());
        cases += 1;
        if let (Ok(Some(filtered)), Ok(Some(exact))) = (&filtered, &exact) {
            decided += 1;
            assert_eq!(filtered, exact, "{kind_x} {kind_y} at {i} and {j}");
        }
        if matches!(exact, Ok(Some(Ordering::Equal)) | Err(_)) {
            assert!(
                !matches!(filtered, Ok(Some(_))),
                "{kind_x} {kind_y} at {i} and {j}"
            );
        }
    }
    assert!(
        decided > cases / 5 && decided < cases * 9 / 10,
        "{decided} of {cases}"
    );
}
