use super::{Derived, Point, decide, raw, side, triple};
use crate::geometry::{
    Coordinate,
    predicates::{arithmetic::Estimate, expansion::Expansion, orient3d, test::Lcg},
};
use core::{array::from_fn, cmp::Ordering};

type Grid = [i128; 3];

fn point(v: Grid) -> Coordinate<3> {
    v.map(|x| x as f64).into()
}

fn minus(a: Grid, b: Grid) -> Grid {
    from_fn(|k| a[k] - b[k])
}

fn plus_scaled(a: Grid, scale: i128, b: Grid) -> Grid {
    from_fn(|k| a[k] + scale * b[k])
}

fn dot(a: Grid, b: Grid) -> i128 {
    (0..3).map(|k| a[k] * b[k]).sum()
}

fn cross(a: Grid, b: Grid) -> Grid {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `(a - d) · ((b - d) × (c - d))`, exactly, which is what `orient3d` is the sign of.
fn orient(a: Grid, b: Grid, c: Grid, d: Grid) -> i128 {
    dot(minus(a, d), cross(minus(b, d), minus(c, d)))
}

fn sign(x: i128) -> Ordering {
    x.cmp(&0)
}

fn random_point(random: &mut Lcg, bound: i128) -> Grid {
    from_fn(|_| (random.bits() as i128).rem_euclid(2 * bound + 1) - bound)
}

/// The side of a plane that the point where a line meets another plane is on, from the point as a
/// fraction with a whole number for a denominator, with the plane scaled to match: the
/// determinant of scaled points is the determinant of the points times the cube of the scale.
fn reference_line_and_plane(
    plane: [Grid; 3],
    line: [Grid; 2],
    other: [Grid; 3],
) -> Option<Ordering> {
    let [p, q] = line;
    let (below_p, below_q) = (
        orient(other[0], other[1], other[2], p),
        orient(other[0], other[1], other[2], q),
    );
    let denominator = below_p - below_q;
    if denominator == 0 {
        return None;
    }
    let numerator = plus_scaled(from_fn(|k| p[k] * denominator), below_p, minus(q, p));
    let scaled = plane.map(|vertex| vertex.map(|x| x * denominator));
    let determinant = orient(scaled[0], scaled[1], scaled[2], numerator);
    Some(sign(determinant * sign(denominator) as i128))
}

/// The same for the point where three planes meet, from Cramer's rule.
fn reference_three_planes(plane: [Grid; 3], planes: [[Grid; 3]; 3]) -> Option<Ordering> {
    let normals = planes.map(|[a, b, c]| cross(minus(b, a), minus(c, a)));
    let offsets: [i128; 3] = from_fn(|i| dot(normals[i], planes[i][0]));
    let determinant = dot(normals[0], cross(normals[1], normals[2]));
    if determinant == 0 {
        return None;
    }
    let crosses = [
        cross(normals[1], normals[2]),
        cross(normals[2], normals[0]),
        cross(normals[0], normals[1]),
    ];
    let numerator: Grid = from_fn(|k| (0..3).map(|i| offsets[i] * crosses[i][k]).sum());
    let scaled = plane.map(|vertex| vertex.map(|x| x * determinant));
    let value = orient(scaled[0], scaled[1], scaled[2], numerator);
    Some(sign(value * sign(determinant) as i128))
}

fn references(plane: [Grid; 3]) -> [Coordinate<3>; 3] {
    plane.map(point)
}

#[test]
fn a_given_point_is_on_the_side_that_orient3d_says() {
    let mut random = Lcg(1);
    for _ in 0..2000 {
        let plane: [Grid; 3] = from_fn(|_| random_point(&mut random, 3));
        let vertex = random_point(&mut random, 3);
        let [a, b, c] = references(plane);
        let found = side([&a, &b, &c], &Derived::Vertex(&point(vertex))).unwrap();
        assert_eq!(found, sign(orient(plane[0], plane[1], plane[2], vertex)));
    }
}

#[test]
fn the_side_of_the_point_where_a_line_meets_a_plane_is_exact() {
    let mut random = Lcg(2);
    let (mut zeros, mut errors) = (0, 0);
    for case in 0..30000 {
        // Coarse grids force zeros and parallel lines; wider ones are generic.
        let bound = [2, 3, 6, 12][case % 4];
        let plane: [Grid; 3] = from_fn(|_| random_point(&mut random, bound));
        let other: [Grid; 3] = if case % 7 == 0 {
            plane
        } else {
            from_fn(|_| random_point(&mut random, bound))
        };
        let line: [Grid; 2] = from_fn(|_| random_point(&mut random, bound));
        let [a, b, c] = references(plane);
        let [u, v, w] = references(other);
        let [p, q] = line.map(point);
        let derived = Derived::Lpi {
            line: [&p, &q],
            plane: [&u, &v, &w],
        };
        let expected = reference_line_and_plane(plane, line, other);
        let found = side([&a, &b, &c], &derived);
        match (expected, found) {
            (Some(expected), Ok(found)) => {
                zeros += (found == Ordering::Equal) as usize;
                assert_eq!(found, expected, "{plane:?} {line:?} {other:?}");
            }
            (None, Err(_)) => errors += 1,
            (expected, found) => panic!("{expected:?} {found:?} {plane:?} {line:?} {other:?}"),
        }
    }
    assert!(
        zeros > 500 && errors > 500,
        "{zeros} zeros, {errors} errors"
    );
}

#[test]
fn the_side_of_the_point_where_three_planes_meet_is_exact() {
    let mut random = Lcg(3);
    let (mut zeros, mut errors) = (0, 0);
    for case in 0..30000 {
        let bound = [2, 3, 5, 8][case % 4];
        let plane: [Grid; 3] = from_fn(|_| random_point(&mut random, bound));
        let planes: [[Grid; 3]; 3] = if case % 5 == 0 {
            // One of the planes is the one being tested.
            [
                plane,
                from_fn(|_| random_point(&mut random, bound)),
                from_fn(|_| random_point(&mut random, bound)),
            ]
        } else {
            from_fn(|_| from_fn(|_| random_point(&mut random, bound)))
        };
        let [a, b, c] = references(plane);
        let owned: [[Coordinate<3>; 3]; 3] = planes.map(references);
        let derived = Derived::Tpi(from_fn(|i| from_fn(|j| &owned[i][j])));
        let expected = reference_three_planes(plane, planes);
        let found = side([&a, &b, &c], &derived);
        match (expected, found) {
            (Some(expected), Ok(found)) => {
                zeros += (found == Ordering::Equal) as usize;
                assert_eq!(found, expected, "{plane:?} {planes:?}");
            }
            (None, Err(_)) => errors += 1,
            (expected, found) => panic!("{expected:?} {found:?} {plane:?} {planes:?}"),
        }
    }
    assert!(
        zeros > 500 && errors > 500,
        "{zeros} zeros, {errors} errors"
    );
}

#[test]
fn a_point_on_a_plane_through_it_is_on_that_plane() {
    // Three planes through one point, and each of them tested at that point.
    let mut random = Lcg(4);
    for _ in 0..2000 {
        let centre = random_point(&mut random, 4);
        let planes: [[Grid; 3]; 3] = from_fn(|_| {
            from_fn(|corner| {
                if corner == 0 {
                    centre
                } else {
                    plus_scaled(centre, 1, random_point(&mut random, 4))
                }
            })
        });
        if reference_three_planes(planes[0], planes).is_none() {
            continue;
        }
        let owned: [[Coordinate<3>; 3]; 3] = planes.map(references);
        let derived = Derived::Tpi(from_fn(|i| from_fn(|j| &owned[i][j])));
        for plane in &owned {
            assert_eq!(
                side([&plane[0], &plane[1], &plane[2]], &derived).unwrap(),
                Ordering::Equal
            );
        }
    }
}

#[test]
fn the_position_of_a_derived_point_is_close_to_the_point() {
    let mut random = Lcg(5);
    let mut checked = 0;
    for case in 0..4000 {
        let bound = [3, 6, 12, 40][case % 4];
        let lines: [Grid; 2] = from_fn(|_| random_point(&mut random, bound));
        let planes: [[Grid; 3]; 3] = from_fn(|_| from_fn(|_| random_point(&mut random, bound)));
        // Where a line meets a plane, as a fraction.
        let [p, q] = lines;
        let (below_p, below_q) = (
            orient(planes[0][0], planes[0][1], planes[0][2], p),
            orient(planes[0][0], planes[0][1], planes[0][2], q),
        );
        let owned: [[Coordinate<3>; 3]; 3] = planes.map(references);
        let [pp, qq] = lines.map(point);
        let lpi = Derived::Lpi {
            line: [&pp, &qq],
            plane: [&owned[0][0], &owned[0][1], &owned[0][2]],
        };
        if below_p != below_q {
            let found = lpi.position().unwrap();
            let fraction = below_p as f64 / (below_p - below_q) as f64;
            for k in 0..3 {
                let expected = p[k] as f64 + fraction * (q[k] - p[k]) as f64;
                assert!((found[k].value() - expected).abs() < 1e-9 * (1.0 + expected.abs()));
            }
            checked += 1;
        } else {
            assert!(lpi.position().is_err());
        }
        let tpi = Derived::Tpi(from_fn(|i| from_fn(|j| &owned[i][j])));
        let normals = planes.map(|[a, b, c]| cross(minus(b, a), minus(c, a)));
        let determinant = dot(normals[0], cross(normals[1], normals[2]));
        if determinant != 0 {
            let offsets: [i128; 3] = from_fn(|i| dot(normals[i], planes[i][0]));
            let crosses = [
                cross(normals[1], normals[2]),
                cross(normals[2], normals[0]),
                cross(normals[0], normals[1]),
            ];
            let found = tpi.position().unwrap();
            for k in 0..3 {
                let numerator: i128 = (0..3).map(|i| offsets[i] * crosses[i][k]).sum();
                let expected = numerator as f64 / determinant as f64;
                assert!((found[k].value() - expected).abs() < 1e-9 * (1.0 + expected.abs()));
            }
            checked += 1;
        } else {
            assert!(tpi.position().is_err());
        }
    }
    assert!(checked > 3000, "{checked}");
}

/// A plane through the point `centre`, from three points around it, none of which is it, with
/// everything scaled by a power of two so that moving a coordinate by one unit in its last place
/// is a different relative change from case to case.
fn plane_through(random: &mut Lcg, centre: Grid, scale: f64) -> [Point; 3] {
    let (first, second) = (random_point(random, 4), random_point(random, 4));
    let third: Grid = from_fn(|k| -first[k] - second[k]);
    [first, second, third].map(|offset| plus_scaled(centre, 1, offset).map(|x| x as f64 * scale))
}

fn nudged(random: &mut Lcg, plane: [Point; 3]) -> [Point; 3] {
    let scale = plane
        .iter()
        .flatten()
        .fold(1e-300_f64, |size, x| size.max(x.abs()));
    plane.map(|vertex| vertex.map(|x| x + random.integer(2) as f64 * scale * f64::EPSILON))
}

fn owned(plane: [Point; 3]) -> [Coordinate<3>; 3] {
    plane.map(|vertex| vertex.into())
}

#[test]
fn the_filter_is_never_certain_of_a_wrong_side() {
    let mut random = Lcg(6);
    let (mut decided, mut undecided) = (0, 0);
    let mut check = |filtered: Option<Ordering>, exact: Ordering| {
        match filtered {
            Some(found) => {
                decided += 1;
                assert_eq!(found, exact);
            }
            None => undecided += 1,
        }
        // An exact zero is never something that the filter claims to know.
        if exact == Ordering::Equal {
            assert!(filtered.is_none());
        }
    };
    for case in 0..20000 {
        let scale = 2.0_f64.powi(random.integer(4) as i32);
        let centre = random_point(&mut random, 3);
        let through = |random: &mut Lcg| plane_through(random, centre, scale);
        // The point is on the plane, up to a few units in the last place in the plane, or the
        // plane is anywhere.
        let near = case % 2 == 0;
        let test = {
            let plane = through(&mut random);
            if near {
                nudged(&mut random, plane)
            } else {
                from_fn(|_| from_fn(|_| (random.unit() - 0.5) * 8.0 * scale))
            }
        };
        let planes: [[Point; 3]; 3] = from_fn(|_| through(&mut random));
        let owned_planes = planes.map(owned);
        let tpi = Derived::Tpi(from_fn(|i| from_fn(|j| &owned_planes[i][j])));
        // A line through the centre, with the centre between its ends.
        let along = random_point(&mut random, 4);
        let ends: [Point; 2] =
            [1, -1].map(|sign| plus_scaled(centre, sign, along).map(|x| x as f64 * scale));
        let lines = ends.map(Coordinate::from);
        let lpi = Derived::Lpi {
            line: [&lines[0], &lines[1]],
            plane: [
                &owned_planes[0][0],
                &owned_planes[0][1],
                &owned_planes[0][2],
            ],
        };
        for point in [&tpi, &lpi] {
            let exact = decide::<Expansion>(test, point);
            let filtered = decide::<Estimate>(test, point);
            match (filtered, exact) {
                (Ok(filtered), Ok(Some(exact))) => check(filtered, exact),
                (Ok(None), Err(_)) => (),
                (Err(_), Err(_)) => (),
                (filtered, exact) => panic!("{filtered:?} {exact:?}"),
            }
        }
    }
    // Rounding settles the cases that are far from the plane, and leaves the others to exact
    // arithmetic.
    assert!(decided > 2000 && undecided > 2000, "{decided} {undecided}");
}

#[test]
fn the_exact_numbers_of_three_planes_stay_short_on_a_grid() {
    let mut random = Lcg(8);
    let mut longest = 0;
    for _ in 0..500 {
        let planes: [[Point; 3]; 3] =
            from_fn(|_| from_fn(|_| random_point(&mut random, 8).map(|x| x as f64)));
        let (numerator, determinant) = triple::<Expansion>(planes, planes[0][0]);
        longest = longest.max(determinant.len());
        numerator
            .iter()
            .for_each(|number| longest = longest.max(number.len()));
    }
    assert!(longest <= 4, "{longest}");
}

#[test]
fn the_exact_numbers_of_three_planes_stay_manageable_on_arbitrary_floats() {
    let mut random = Lcg(9);
    let mut longest = 0;
    for _ in 0..200 {
        let planes: [[Point; 3]; 3] =
            from_fn(|_| from_fn(|_| from_fn(|_| random.unit() * 10.0 - 5.0)));
        let (numerator, determinant) = triple::<Expansion>(planes, planes[0][0]);
        longest = longest.max(determinant.len());
        numerator
            .iter()
            .for_each(|number| longest = longest.max(number.len()));
    }
    // Compressing keeps these to the number of bits they hold, over 53, not what they were
    // grown through.
    assert!(longest <= 60, "{longest}");
    let _ = raw(&Coordinate::from([0.0; 3]));
    let _ = orient3d::value::<Estimate>([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]);
}
