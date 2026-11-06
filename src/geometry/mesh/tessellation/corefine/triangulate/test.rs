use super::{Across, Input, Triangulation, build, triangulate};
use crate::math::{FxHashMap, FxHashSet};

struct Lcg(u64);

impl Lcg {
    fn bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }
    fn chance(&mut self, probability: f64) -> bool {
        (self.bits() as f64 / (1u64 << 53) as f64) < probability
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

struct Case {
    input: Input,
    integer: Vec<[i64; 2]>,
    gram: [i64; 3],
    size: i64,
}

fn make(random: &mut Lcg, size: i64, sides: f64, interior: f64, gram: [i64; 3]) -> Case {
    let mut found: Vec<(usize, [i64; 2])> = Vec::new();
    for t in 1..size {
        if random.chance(sides) {
            found.push((0, [t, 0]));
        }
    }
    for t in 1..size {
        if random.chance(sides) {
            found.push((1, [size - t, t]));
        }
    }
    for t in 1..size {
        if random.chance(sides) {
            found.push((2, [0, size - t]));
        }
    }
    for x in 1..size {
        for y in 1..size - x {
            if random.chance(interior) {
                found.push((3, [x, y]));
            }
        }
    }
    let mut ranks: Vec<usize> = (0..found.len()).collect();
    random.shuffle(&mut ranks);
    let mut integer = vec![[0, 0], [size, 0], [0, size]];
    integer.resize(3 + found.len(), [0, 0]);
    let mut side_points: [Vec<usize>; 3] = [vec![], vec![], vec![]];
    for (k, &(group, point)) in found.iter().enumerate() {
        let index = 3 + ranks[k];
        integer[index] = point;
        if group < 3 {
            side_points[group].push(index);
        }
    }
    Case {
        input: Input {
            gram: gram.map(|g| g as f64),
            points: integer.iter().map(|p| p.map(|x| x as f64)).collect(),
            sides: side_points,
            segments: Vec::new(),
        },
        integer,
        gram,
        size,
    }
}

fn orient(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i128 {
    (a[0] - c[0]) as i128 * (b[1] - c[1]) as i128 - (a[1] - c[1]) as i128 * (b[0] - c[0]) as i128
}

fn cross(case: &Case, [a, b]: [usize; 2], [c, d]: [usize; 2]) -> bool {
    let p = |i: usize| case.integer[i];
    orient(p(a), p(b), p(c)).signum() * orient(p(a), p(b), p(d)).signum() < 0
        && orient(p(c), p(d), p(a)).signum() * orient(p(c), p(d), p(b)).signum() < 0
}

fn with_segments(case: &mut Case, random: &mut Lcg, wanted: usize) {
    let count = case.integer.len();
    let mut accepted: Vec<[usize; 2]> = Vec::new();
    for _ in 0..4 * wanted {
        if accepted.len() == wanted {
            break;
        }
        let candidate = [random.below(count), random.below(count)];
        if candidate[0] != candidate[1]
            && !accepted.iter().any(|&other| cross(case, other, candidate))
        {
            accepted.push(candidate);
        }
    }
    case.input.segments = accepted;
}

fn incircle(case: &Case, a: usize, b: usize, c: usize, d: usize) -> i128 {
    let point = |i: usize| case.integer[i];
    let [g0, g1, g2] = case.gram.map(|g| g as i128);
    let to = |i: usize| -> [i128; 2] {
        [
            (point(i)[0] - point(d)[0]) as i128,
            (point(i)[1] - point(d)[1]) as i128,
        ]
    };
    let lift = |[x, y]: [i128; 2]| g0 * x * x + 2 * g1 * x * y + g2 * y * y;
    let ([ax, ay], [bx, by], [cx, cy]) = (to(a), to(b), to(c));
    lift([ax, ay]) * (bx * cy - cx * by)
        + lift([bx, by]) * (cx * ay - ax * cy)
        + lift([cx, cy]) * (ax * by - bx * ay)
}

fn segment_edges(case: &Case) -> FxHashSet<(usize, usize)> {
    let p = |i: usize| case.integer[i];
    let mut edges = FxHashSet::default();
    for &[u, v] in &case.input.segments {
        let axis = if p(u)[0] != p(v)[0] { 0 } else { 1 };
        let (low, high) = (p(u)[axis].min(p(v)[axis]), p(u)[axis].max(p(v)[axis]));
        let mut along: Vec<usize> = (0..case.integer.len())
            .filter(|&i| orient(p(u), p(v), p(i)) == 0 && low <= p(i)[axis] && p(i)[axis] <= high)
            .collect();
        along.sort_by_key(|&i| p(i)[axis]);
        along.windows(2).for_each(|w| {
            edges.insert((w[0].min(w[1]), w[0].max(w[1])));
        });
    }
    edges
}

fn verify(case: &Case, triangles: &[[usize; 3]]) {
    let point = |i: usize| case.integer[i];
    let count = case.integer.len();
    let boundary_points: usize = case.input.sides.iter().map(Vec::len).sum::<usize>() + 3;
    assert_eq!(triangles.len(), 2 * count - boundary_points - 2);
    let mut twice_area = 0;
    let mut third: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for &[a, b, c] in triangles {
        let area = orient(point(a), point(b), point(c));
        assert!(area > 0, "triangle {a} {b} {c} is not counterclockwise");
        twice_area += area;
        for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)] {
            assert!(
                third.insert((u, v), w).is_none(),
                "edge {u} {v} is used twice"
            );
        }
    }
    assert_eq!(twice_area, (case.size * case.size) as i128);
    let boundary: FxHashSet<(usize, usize)> = third
        .keys()
        .copied()
        .filter(|&(u, v)| !third.contains_key(&(v, u)))
        .collect();
    let mut expected = FxHashSet::default();
    for (edge, side) in case.input.sides.iter().enumerate() {
        let mut chain = vec![edge];
        chain.extend(side);
        chain.push((edge + 1) % 3);
        chain.windows(2).for_each(|w| {
            expected.insert((w[0], w[1]));
        });
    }
    assert_eq!(boundary, expected);
    let constrained = segment_edges(case);
    for &(u, v) in &constrained {
        assert!(
            third.contains_key(&(u, v)) || third.contains_key(&(v, u)),
            "the segment edge {u} {v} is missing"
        );
    }
    for &[a, b, c] in triangles {
        if case.input.segments.is_empty() {
            for d in (0..count).filter(|d| ![a, b, c].contains(d)) {
                assert!(
                    incircle(case, a, b, c, d) <= 0,
                    "{d} is inside the circle of {a} {b} {c}"
                );
            }
        }
        for (x, y, z) in [(a, b, c), (b, c, a), (c, a, b)] {
            if constrained.contains(&(y.min(z), y.max(z))) {
                continue;
            }
            if let Some(&d) = third.get(&(z, y)) {
                let inside = incircle(case, x, y, z, d);
                assert!(inside <= 0, "{d} is inside the circle of {x} {y} {z}");
                if inside == 0 {
                    let lowest = x.min(y).min(z).min(d);
                    assert!(
                        lowest != y && lowest != z,
                        "the tie at {x} {y} {z} {d} is wrong"
                    );
                }
            }
        }
    }
}

fn consistent(triangulation: &Triangulation) {
    for (f, face) in triangulation.faces.iter().enumerate() {
        for i in 0..3 {
            let (u, v) = (face.nodes[(i + 1) % 3], face.nodes[(i + 2) % 3]);
            if let Across::Face(g) = face.across[i] {
                let other = &triangulation.faces[g];
                let j = (0..3)
                    .find(|&j| other.nodes[(j + 1) % 3] == v && other.nodes[(j + 2) % 3] == u)
                    .expect("the neighbour does not share the edge");
                assert_eq!(other.across[j], Across::Face(f));
            }
        }
    }
}

fn check_all(case: &Case, random: &mut Lcg) {
    let triangulation = build(&case.input, [0, 1, 2], &case.input.interior()).unwrap();
    consistent(&triangulation);
    let triangles = triangulation.triangles();
    verify(case, &triangles);
    for _ in 0..4 {
        let mut edges = [0, 1, 2];
        random.shuffle(&mut edges);
        let mut interior = case.input.interior();
        random.shuffle(&mut interior);
        let mut input = case.input.clone();
        random.shuffle(&mut input.segments);
        input.segments.iter_mut().for_each(|segment| {
            if random.chance(0.5) {
                segment.reverse()
            }
        });
        let shuffled = build(&input, edges, &interior).unwrap();
        consistent(&shuffled);
        assert_eq!(
            shuffled.triangles(),
            triangles,
            "the order of insertion changed the result"
        );
    }
}

const GRAMS: [[i64; 3]; 4] = [[1, 0, 1], [2, 1, 3], [1, -1, 2], [3, 0, 1]];

#[test]
fn a_triangle_alone_is_one_triangle() {
    let case = make(&mut Lcg(1), 5, 0.0, 0.0, [1, 0, 1]);
    assert_eq!(triangulate(&case.input).unwrap(), vec![[0, 1, 2]]);
}

#[test]
fn points_on_the_edges_alone_make_a_valid_triangulation() {
    let mut random = Lcg(2);
    (0..300).for_each(|n| {
        let case = make(&mut random, 3 + n % 12, 0.6, 0.0, GRAMS[n as usize % 4]);
        check_all(&case, &mut random);
    })
}

#[test]
fn random_points_make_a_valid_delaunay_triangulation() {
    let mut random = Lcg(3);
    (0..600).for_each(|n| {
        let size = 3 + n % 14;
        let case = make(
            &mut random,
            size,
            0.5,
            0.15 + 0.1 * (n % 7) as f64,
            GRAMS[n as usize % 4],
        );
        check_all(&case, &mut random);
    })
}

#[test]
fn a_full_lattice_is_cocircular_everywhere_and_the_result_is_still_unique() {
    let mut random = Lcg(4);
    (0..40).for_each(|n| {
        let case = make(&mut random, 4 + (n % 9), 1.0, 1.0, GRAMS[n as usize % 4]);
        check_all(&case, &mut random);
    })
}

#[test]
fn a_segment_across_the_triangle_is_an_edge() {
    let mut random = Lcg(5);
    let mut case = make(&mut random, 6, 1.0, 1.0, [1, 0, 1]);
    let find = |at: [i64; 2]| case.integer.iter().position(|&p| p == at).unwrap();
    case.input.segments = vec![[find([5, 0]), find([0, 5])]];
    case.input.segments.push([find([1, 1]), find([1, 3])]);
    check_all(&case, &mut random);
}

#[test]
fn random_segments_are_recovered_and_everything_else_is_delaunay() {
    let mut random = Lcg(6);
    (0..400).for_each(|n| {
        let size = 3 + n % 12;
        let mut case = make(
            &mut random,
            size,
            0.5,
            0.2 + 0.1 * (n % 6) as f64,
            GRAMS[n as usize % 4],
        );
        let wanted = 1 + random.below(8);
        with_segments(&mut case, &mut random, wanted);
        check_all(&case, &mut random);
    })
}

#[test]
fn segments_through_lattice_points_and_along_the_boundary_are_joined_through_them() {
    let mut random = Lcg(7);
    (0..40).for_each(|n| {
        let mut case = make(&mut random, 4 + (n % 9), 1.0, 1.0, GRAMS[n as usize % 4]);
        let size = case.size;
        let find = |at: [i64; 2]| case.integer.iter().position(|&p| p == at).unwrap();
        case.input.segments = vec![
            [0, 1],
            [1, 2],
            [2, 0],
            [find([1, 0]), find([size - 1, 0])],
            [find([1, 1]), find([size - 2, 1])],
            [0, find([size - 2, 1])],
        ];
        with_segments(&mut case, &mut random, 3);
        check_all(&case, &mut random);
    })
}

#[test]
fn many_segments_in_a_full_lattice_are_all_recovered() {
    let mut random = Lcg(8);
    (0..30).for_each(|n| {
        let mut case = make(&mut random, 6 + (n % 8), 1.0, 1.0, GRAMS[n as usize % 4]);
        with_segments(&mut case, &mut random, 25);
        check_all(&case, &mut random);
    })
}

#[test]
fn crossing_segments_are_an_error() {
    let mut random = Lcg(9);
    let mut case = make(&mut random, 6, 1.0, 1.0, [1, 0, 1]);
    let find = |at: [i64; 2]| case.integer.iter().position(|&p| p == at).unwrap();
    case.input.segments = vec![[find([1, 1]), find([3, 2])], [find([2, 0]), find([2, 3])]];
    assert!(triangulate(&case.input).is_err());
    case.input.segments = vec![[find([1, 1]), find([3, 1])], [find([2, 0]), find([2, 3])]];
    assert!(triangulate(&case.input).is_ok());
}

#[test]
fn bad_input_is_an_error() {
    let case = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    let mut clockwise = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    clockwise.input.points.swap(1, 2);
    assert!(triangulate(&clockwise.input).is_err());
    let mut doubled = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    let copy = doubled.input.points[3];
    doubled.input.points.push(copy);
    assert!(triangulate(&doubled.input).is_err());
    let mut on_the_edge = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    on_the_edge.input.points.push([3.0, 0.0]);
    assert!(triangulate(&on_the_edge.input).is_err());
    let mut not_on_the_edge = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    not_on_the_edge.input.points.push([1.0, 1.0]);
    let last = not_on_the_edge.input.points.len() - 1;
    not_on_the_edge.input.sides[0].push(last);
    assert!(triangulate(&not_on_the_edge.input).is_err());
    let mut no_length = make(&mut Lcg(5), 6, 0.0, 0.5, [1, 0, 1]);
    no_length.input.segments = vec![[3, 3]];
    assert!(triangulate(&no_length.input).is_err());
    assert!(triangulate(&case.input).is_ok());
}
