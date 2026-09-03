//! automesh #757: is the near-collinear "two edges on one crease" hex the
//! dominant cause of the low-MSJ tail on non-axis-aligned creases?
//!
//! `cargo test -F geometry -- --nocapture --ignored buffer::diagnose`

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Fitting, Mesh, Verdict, tessellation::Tessellation},
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::array::from_fn;

const HEX_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// A hex edge lies on a crease if both ends are within `CREASE_TOL * h` of one
/// crease segment and the edge runs within `ALONG` of it.
const CREASE_TOL: Scalar = 0.45;
/// cos(20 deg): an edge counts as lying along a crease below this angle.
const ALONG: Scalar = 0.939_692_6;
/// cos(150 deg): a shared-vertex pair this straight has a collapsing corner.
const STRAIGHT: Scalar = -0.866_025_4;

fn xyz(point: &Coordinate<3>) -> [Scalar; 3] {
    from_fn(|i| point[i].value())
}

fn sub(a: [Scalar; 3], b: [Scalar; 3]) -> [Scalar; 3] {
    from_fn(|i| a[i] - b[i])
}

fn dot(a: [Scalar; 3], b: [Scalar; 3]) -> Scalar {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [Scalar; 3]) -> Scalar {
    dot(a, a).sqrt()
}

/// Distance from `p` to segment `[a, b]`, and the unit segment direction.
fn to_segment(p: [Scalar; 3], a: [Scalar; 3], b: [Scalar; 3]) -> (Scalar, [Scalar; 3]) {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    let t = if len2 > 0.0 {
        (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let foot = from_fn(|i| a[i] + t * ab[i]);
    let dir = if len2 > 0.0 {
        from_fn(|i| ab[i] / len2.sqrt())
    } else {
        [0.0; 3]
    };
    (norm(sub(p, foot)), dir)
}

/// Groups crease segments into feature lines: segments that share an endpoint
/// and run within 12 deg of each other are the same line, so a polyline that
/// turns a corner splits in two.
fn crease_lines(creases: &[[Coordinate<3>; 2]], eps: Scalar) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..creases.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let dir = |[a, b]: &[Coordinate<3>; 2]| {
        let d = sub(xyz(b), xyz(a));
        let n = norm(d);
        from_fn::<Scalar, 3, _>(|i| if n > 0.0 { d[i] / n } else { 0.0 })
    };
    let key = |c: [Scalar; 3]| -> [i64; 3] { from_fn(|i| (c[i] / eps).round() as i64) };
    let mut seen: std::collections::HashMap<[i64; 3], Vec<usize>> =
        std::collections::HashMap::new();
    for (index, [a, b]) in creases.iter().enumerate() {
        for endpoint in [xyz(a), xyz(b)] {
            seen.entry(key(endpoint)).or_default().push(index)
        }
    }
    for members in seen.values() {
        for (m, &i) in members.iter().enumerate() {
            for &j in &members[m + 1..] {
                if dot(dir(&creases[i]), dir(&creases[j])).abs() > 0.978 {
                    let (x, y) = (find(&mut parent, i), find(&mut parent, j));
                    parent[x] = y
                }
            }
        }
    }
    (0..creases.len()).map(|i| find(&mut parent, i)).collect()
}

struct HexScan {
    /// Two edges on one feature line, sharing a vertex, near collinear.
    collapsing_corner: bool,
    /// Two or more edges on one feature line, shared vertex or not.
    two_on_one_line: bool,
}

fn scan_hex(
    hex: &[usize; 8],
    coordinates: &Coordinates<3>,
    creases: &[[Coordinate<3>; 2]],
    line_of: &[usize],
) -> HexScan {
    let p: [[Scalar; 3]; 8] = from_fn(|i| xyz(&coordinates[hex[i]]));
    let h = HEX_EDGES
        .iter()
        .map(|&[i, j]| norm(sub(p[i], p[j])))
        .sum::<Scalar>()
        / 12.0;
    let tol = CREASE_TOL * h;
    // Feature line each hex edge lies along, if any.
    let mut edge_line = [None::<usize>; 12];
    for (slot, &[i, j]) in edge_line.iter_mut().zip(HEX_EDGES.iter()) {
        let edge = sub(p[j], p[i]);
        let edge_len = norm(edge);
        if edge_len == 0.0 {
            continue;
        }
        let edge_dir = from_fn(|k| edge[k] / edge_len);
        let mut best: Option<(Scalar, usize)> = None;
        for (index, [a, b]) in creases.iter().enumerate() {
            let (a, b) = (xyz(a), xyz(b));
            let (di, seg_dir) = to_segment(p[i], a, b);
            let (dj, _) = to_segment(p[j], a, b);
            if di < tol && dj < tol && dot(edge_dir, seg_dir).abs() > ALONG {
                let far = di.max(dj);
                if best.is_none_or(|(d, _)| far < d) {
                    best = Some((far, line_of[index]))
                }
            }
        }
        *slot = best.map(|(_, line)| line);
    }
    let mut two_on_one_line = false;
    for a in 0..12 {
        for b in (a + 1)..12 {
            if let (Some(la), Some(lb)) = (edge_line[a], edge_line[b])
                && la == lb
            {
                two_on_one_line = true
            }
        }
    }
    // A near-straight pair of on-line edges through a shared hex vertex.
    let mut collapsing_corner = false;
    for a in 0..12 {
        for b in (a + 1)..12 {
            let (Some(la), Some(lb)) = (edge_line[a], edge_line[b]) else {
                continue;
            };
            if la != lb {
                continue;
            }
            let (ea, eb) = (HEX_EDGES[a], HEX_EDGES[b]);
            let shared = [ea[0], ea[1]].into_iter().find(|v| eb.contains(v));
            let Some(shared) = shared else { continue };
            let oa = if ea[0] == shared { ea[1] } else { ea[0] };
            let ob = if eb[0] == shared { eb[1] } else { eb[0] };
            let va = sub(p[oa], p[shared]);
            let vb = sub(p[ob], p[shared]);
            let (na, nb) = (norm(va), norm(vb));
            if na > 0.0 && nb > 0.0 && dot(va, vb) / (na * nb) < STRAIGHT {
                collapsing_corner = true
            }
        }
    }
    HexScan {
        collapsing_corner,
        two_on_one_line,
    }
}

fn median(mut values: Vec<Scalar>) -> Scalar {
    if values.is_empty() {
        return Scalar::NAN;
    }
    values.sort_by(Scalar::total_cmp);
    values[values.len() / 2]
}

fn report(name: &str, target: &Tessellation, scale: Scalar) {
    let mut octree =
        Octree::<u32, usize>::from_features(target, scale, CurvatureSizing::default(), 0)
            .expect("octree");
    octree
        .equilibrate(Balancing::Strong(1), Pairing::Regular)
        .expect("equilibrate");
    let mut mesh = octree.dualize();
    target.trim(&mut mesh).expect("trim");
    let mesh = mesh.buffer(target, Fitting::Snap).expect("buffer");

    let creases = target.features().creases();
    let bbox = {
        let c = mesh.coordinates();
        let mut lo = [Scalar::INFINITY; 3];
        let mut hi = [Scalar::NEG_INFINITY; 3];
        for point in c.iter() {
            for k in 0..3 {
                lo[k] = lo[k].min(point[k].value());
                hi[k] = hi[k].max(point[k].value());
            }
        }
        norm(sub(hi, lo))
    };
    let line_of = crease_lines(creases, 1.0e-6 * bbox);

    let coordinates = mesh.coordinates();
    let msj_blocks = mesh.minimum_scaled_jacobians();
    let mut rows: Vec<(Scalar, HexScan)> = Vec::new();
    for (block, msj) in mesh.iter().zip(msj_blocks.iter()) {
        if let Connectivity::Hexahedral(hexes) = block {
            for (hex, &quality) in hexes.iter().zip(msj.iter()) {
                let hex: [usize; 8] = from_fn(|i| hex[i]);
                rows.push((quality, scan_hex(&hex, coordinates, creases, &line_of)));
            }
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n = rows.len();
    let corner = |r: &(Scalar, HexScan)| r.1.collapsing_corner;
    let flagged = rows.iter().filter(|r| corner(r)).count();
    let two_line = rows.iter().filter(|r| r.1.two_on_one_line).count();
    let split = |pred: &dyn Fn(&(Scalar, HexScan)) -> bool| {
        (
            median(rows.iter().filter(|r| pred(r)).map(|r| r.0).collect()),
            rows.iter()
                .filter(|r| pred(r))
                .map(|r| r.0)
                .fold(Scalar::INFINITY, Scalar::min),
        )
    };
    let (med_f, min_f) = split(&|r| corner(r));
    let (med_u, min_u) = split(&|r| !corner(r));
    let worst10 = rows.iter().take(10).filter(|r| corner(r)).count();
    let below = |t: Scalar| {
        let hit: Vec<_> = rows.iter().filter(|r| r.0 < t).collect();
        let f = hit.iter().filter(|r| corner(r)).count();
        (hit.len(), f)
    };
    let (lt10, lt10_f) = below(0.1);
    let (lt20, lt20_f) = below(0.2);

    println!(
        "\n{name:<20} hexes {n:<5} creases {:<5} lines {:<4} collapsing-corner {flagged:<4} two-on-line {two_line}",
        creases.len(),
        line_of
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
    );
    println!(
        "  MSJ min {:.4}   flagged: min {min_f:.4} median {med_f:.4}   clean: min {min_u:.4} median {med_u:.4}",
        rows.first().map_or(Scalar::NAN, |r| r.0),
    );
    println!("  worst 10 by MSJ that are flagged: {worst10}/10");
    println!(
        "  MSJ < 0.1: {lt10:<4} of which flagged {lt10_f}      MSJ < 0.2: {lt20:<4} of which flagged {lt20_f}"
    );
}

fn surface(points: Vec<[Scalar; 3]>, faces: Vec<[usize; 3]>) -> Tessellation {
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

/// Extrudes a counterclockwise polygon into a closed solid, fan-capped from
/// `kernel`, which must see every vertex.
fn extrude(
    polygon: &[[Scalar; 2]],
    kernel: [Scalar; 2],
    bottom: Scalar,
    top: Scalar,
) -> Tessellation {
    let n = polygon.len();
    let mut points = Vec::with_capacity(2 * n + 2);
    polygon
        .iter()
        .for_each(|&[x, y]| points.push([x, y, bottom]));
    polygon.iter().for_each(|&[x, y]| points.push([x, y, top]));
    points.push([kernel[0], kernel[1], bottom]);
    points.push([kernel[0], kernel[1], top]);
    let mut faces = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        faces.push([2 * n, j, i]);
        faces.push([2 * n + 1, n + i, n + j]);
        faces.push([i, j, n + j]);
        faces.push([i, n + j, n + i]);
    }
    surface(points, faces)
}

/// Rigid rotation by `angle` about two axes, taking every crease off-grid.
fn rotate(tessellation: &Tessellation, angle: Scalar) -> Tessellation {
    let (sine, cosine) = angle.sin_cos();
    let mesh = tessellation.mesh();
    let points: Vec<[Scalar; 3]> = mesh
        .coordinates()
        .iter()
        .map(|point| {
            let (x, y, z) = (point[0].value(), point[1].value(), point[2].value());
            let (u, v) = (cosine * x - sine * y, sine * x + cosine * y);
            let (w, t) = (cosine * v - sine * z, sine * v + cosine * z);
            [u, w, t]
        })
        .collect();
    let faces: Vec<[usize; 3]> = mesh
        .iter()
        .flatten()
        .map(|face| from_fn(|i| face[i]))
        .collect();
    surface(points, faces)
}

fn staircase() -> Tessellation {
    extrude(
        &[
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [0.0, 3.0],
        ],
        [0.5, 0.5],
        0.0,
        1.5,
    )
}

/// A concave feature: a rectangular slot cut into one face.
fn notched() -> Tessellation {
    extrude(
        &[
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ],
        [0.5, 0.5],
        0.0,
        1.5,
    )
}

#[test]
#[ignore = "diagnostic, run explicitly with --nocapture"]
fn near_collinear_hex_dominates_the_low_msj_tail() {
    println!();
    report("staircase axis", &staircase(), 5.0);
    report("staircase 25deg", &rotate(&staircase(), 0.436_332), 5.0);
    report("staircase 40deg", &rotate(&staircase(), 0.698_132), 5.0);
    report("notched axis", &notched(), 5.0);
    report("notched 25deg", &rotate(&notched(), 0.436_332), 5.0);
}

/// Same scan on the bone, a real creased input. Needs `target/bone_tri.stl`
/// (copy it from the automesh repo); skips if absent.
#[test]
#[ignore = "diagnostic, needs target/bone_tri.stl"]
fn near_collinear_hex_on_the_bone() {
    let path = std::path::Path::new("target/bone_tri.stl");
    if !path.exists() {
        println!("\nskipped: {} not found", path.display());
        return;
    }
    let bone = Tessellation::try_from(path).expect("read bone");
    println!();
    report("bone axis", &bone, 4.0);
    report("bone 25deg", &rotate(&bone, 0.436_332), 4.0);
}
