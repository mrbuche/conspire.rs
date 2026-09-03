//! A free-form face — a B-spline / NURBS surface, or a surface of revolution —
//! as a sampled parametric patch. Neither has a closed-form point inversion or
//! ray intersection, so both are handled numerically: a dense `(u, v)` sample
//! grid seeds a Gauss-Newton closest-point solve, and the same grid,
//! triangulated, is the ray-hit target for the parity sign. The grid is an
//! evaluation cache, not mesh output. Trimming is a `(u, v)` polygon walked
//! from the 3D edges through the point-inversion, the same chorded-edge
//! treatment a spline edge already gets on a cylinder.

use super::super::{
    Brep, D, Face, Loop,
    curve::Curve,
    surface::{BSplineSurface, Revolution},
};
use super::{Ring, chart_contains, chart_nearest, sweeps_whole_surface, wrap_by};
use crate::{geometry::Coordinate, math::Scalar};
use std::array::from_fn;
use std::f64::consts::TAU;

const RAY_EPS: Scalar = 1.0e-12;
/// Gauss-Newton point-inversion iterations.
const NEWTON: usize = 20;

/// A parametric surface `S(u, v)` evaluated numerically.
enum Field {
    /// A tensor-product B-spline in homogeneous control points, indexed
    /// `control[iu][iv]` as STEP stores it.
    BSpline {
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<Scalar>,
        v_knots: Vec<Scalar>,
        control: Vec<Vec<[Scalar; D + 1]>>,
    },
    /// A profile polyline (`v` its normalized arc parameter) revolved about the
    /// line through `origin` along `axis` (`u` the angle).
    Revolution {
        profile: Vec<[Scalar; D]>,
        origin: [Scalar; D],
        axis: [Scalar; D],
    },
}

impl Field {
    fn point(&self, u: Scalar, v: Scalar) -> [Scalar; D] {
        match self {
            Self::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                control,
            } => {
                let column: Vec<[Scalar; D + 1]> = control
                    .iter()
                    .map(|row| de_boor(*v_degree, v_knots, row, v))
                    .collect();
                let homogeneous = de_boor(*u_degree, u_knots, &column, u);
                let w = if homogeneous[D].abs() > Scalar::EPSILON {
                    homogeneous[D]
                } else {
                    1.0
                };
                from_fn(|k| homogeneous[k] / w)
            }
            Self::Revolution {
                profile,
                origin,
                axis,
            } => {
                let last = profile.len() - 1;
                let s = v.clamp(0.0, 1.0) * last as Scalar;
                let i = (s.floor() as usize).min(last - 1);
                let f = s - i as Scalar;
                let p: [Scalar; D] = from_fn(|k| profile[i][k] * (1.0 - f) + profile[i + 1][k] * f);
                rotate_about(p, *origin, *axis, u)
            }
        }
    }
}

/// `p` rotated by `angle` about the line through `origin` along the unit
/// vector `axis` (Rodrigues).
fn rotate_about(
    p: [Scalar; D],
    origin: [Scalar; D],
    axis: [Scalar; D],
    angle: Scalar,
) -> [Scalar; D] {
    let r: [Scalar; D] = from_fn(|k| p[k] - origin[k]);
    let (c, s) = (angle.cos(), angle.sin());
    let along = dot(axis, r);
    let cross_kr = cross(axis, r);
    from_fn(|k| origin[k] + r[k] * c + cross_kr[k] * s + axis[k] * along * (1.0 - c))
}

/// De Boor evaluation of a homogeneous B-spline at `t`, clamped to the span.
fn de_boor(
    degree: usize,
    knots: &[Scalar],
    control: &[[Scalar; D + 1]],
    t: Scalar,
) -> [Scalar; D + 1] {
    let count = control.len();
    let degree = degree.min(count.saturating_sub(1));
    let (low, high) = (knots[degree], knots[count]);
    let t = t.clamp(low, high);
    let mut span = degree;
    while span + 1 < count && knots[span + 1] <= t {
        span += 1;
    }
    let mut work: Vec<[Scalar; D + 1]> = (0..=degree).map(|j| control[j + span - degree]).collect();
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let i = j + span - degree;
            let (lower, upper) = (knots[i], knots[i + degree + 1 - r]);
            let alpha = if upper > lower {
                (t - lower) / (upper - lower)
            } else {
                0.0
            };
            let previous = work[j - 1];
            for k in 0..=D {
                work[j][k] = (1.0 - alpha) * previous[k] + alpha * work[j][k];
            }
        }
    }
    work[degree]
}

pub(super) struct Sampled {
    field: Field,
    nu: usize,
    nv: usize,
    u_range: [Scalar; 2],
    v_range: [Scalar; 2],
    u_period: Option<Scalar>,
    v_period: Option<Scalar>,
    /// `nu * nv` surface points, `grid[iu * nv + iv]`.
    grid: Vec<[Scalar; D]>,
    /// Triangles into `grid`, the ray-hit target.
    triangles: Vec<[usize; 3]>,
    /// `(u, v)` trim polygons; `None` when the face sweeps the whole surface.
    rings: Option<Vec<Ring>>,
    /// Mean `|dS/du|`, the arc-length weight for the trim's nearest-boundary snap.
    u_weight: Scalar,
    sign: Scalar,
    tolerance: Scalar,
    diagonal: Scalar,
    pub(super) low: [Scalar; D],
    pub(super) high: [Scalar; D],
}

impl Sampled {
    fn u_at(&self, iu: usize) -> Scalar {
        self.u_range[0]
            + (self.u_range[1] - self.u_range[0]) * iu as Scalar / (self.nu - 1) as Scalar
    }

    fn v_at(&self, iv: usize) -> Scalar {
        self.v_range[0]
            + (self.v_range[1] - self.v_range[0]) * iv as Scalar / (self.nv - 1) as Scalar
    }

    fn grid_uv(&self, index: usize) -> [Scalar; 2] {
        [self.u_at(index / self.nv), self.v_at(index % self.nv)]
    }

    /// `(point, outward unit normal)` at `(u, v)`, partials by central difference.
    fn point_normal(&self, [u, v]: [Scalar; 2]) -> ([Scalar; D], [Scalar; D]) {
        let point = self.field.point(u, v);
        let hu = (self.u_range[1] - self.u_range[0]) * 1.0e-5;
        let hv = (self.v_range[1] - self.v_range[0]) * 1.0e-5;
        let su = diff(
            self.field.point(u + hu, v),
            self.field.point(u - hu, v),
            2.0 * hu,
        );
        let sv = diff(
            self.field.point(u, v + hv),
            self.field.point(u, v - hv),
            2.0 * hv,
        );
        let normal = unit(cross(su, sv))
            .or_else(|| unit(su))
            .or_else(|| unit(sv))
            .unwrap_or([1.0, 0.0, 0.0]);
        (point, normal.map(|x| self.sign * x))
    }

    /// The `(u, v)` whose surface point is nearest `q`: nearest grid sample,
    /// then a damped Gauss-Newton descent of `|S(u, v) - q|^2`.
    fn invert(&self, q: [Scalar; D]) -> [Scalar; 2] {
        let mut seed = 0;
        let mut best = Scalar::INFINITY;
        for (index, sample) in self.grid.iter().enumerate() {
            let d = (0..D).map(|k| (sample[k] - q[k]).powi(2)).sum::<Scalar>();
            if d < best {
                best = d;
                seed = index;
            }
        }
        let [mut u, mut v] = self.grid_uv(seed);
        let hu = (self.u_range[1] - self.u_range[0]) * 1.0e-5;
        let hv = (self.v_range[1] - self.v_range[0]) * 1.0e-5;
        for _ in 0..NEWTON {
            let s = self.field.point(u, v);
            let su = diff(
                self.field.point(u + hu, v),
                self.field.point(u - hu, v),
                2.0 * hu,
            );
            let sv = diff(
                self.field.point(u, v + hv),
                self.field.point(u, v - hv),
                2.0 * hv,
            );
            let r: [Scalar; D] = from_fn(|k| s[k] - q[k]);
            let (guu, guv, gvv) = (dot(su, su), dot(su, sv), dot(sv, sv));
            let (bu, bv) = (dot(su, r), dot(sv, r));
            // Levenberg damping keeps the 2x2 solve well posed at a flat spot.
            let lambda = 1.0e-9 * (guu + gvv).max(1.0);
            let (a, d_) = (guu + lambda, gvv + lambda);
            let determinant = a * d_ - guv * guv;
            if determinant.abs() < 1.0e-30 {
                break;
            }
            let step_u = (d_ * bu - guv * bv) / determinant;
            let step_v = (a * bv - guv * bu) / determinant;
            u -= step_u;
            v -= step_v;
            u = clamp_or_wrap(u, self.u_range, self.u_period);
            v = clamp_or_wrap(v, self.v_range, self.v_period);
            if step_u.abs() <= hu && step_v.abs() <= hv {
                break;
            }
        }
        [u, v]
    }

    /// Closest point on the trimmed patch to `q`, and its outward unit normal.
    pub(super) fn closest(&self, q: [Scalar; D]) -> ([Scalar; D], [Scalar; D]) {
        let uv = self.invert(q);
        let uv = match &self.rings {
            None => uv,
            Some(rings) if chart_contains(uv, rings, self.v_period) => uv,
            Some(rings) => {
                let weight = self.u_weight;
                chart_nearest(uv, rings, move |_| weight, self.v_period)
            }
        };
        self.point_normal(uv)
    }

    /// Ray parameters `t > 0` where `o + t*d` crosses the trimmed patch, and
    /// whether any landed within the trim tolerance of the boundary.
    pub(super) fn ray_hits(&self, o: [Scalar; D], d: [Scalar; D]) -> (Vec<Scalar>, bool) {
        let tolerance = self.tolerance;
        let mut hits: Vec<Scalar> = Vec::new();
        let mut grazed = false;
        for &[ia, ib, ic] in &self.triangles {
            let Some((t, bary)) = intersect(o, d, self.grid[ia], self.grid[ib], self.grid[ic])
            else {
                continue;
            };
            if t <= RAY_EPS {
                continue;
            }
            let uv = blend(self.grid_uv(ia), self.grid_uv(ib), self.grid_uv(ic), bary);
            if let Some(rings) = &self.rings {
                if tolerance > 0.0 {
                    let near = chart_nearest(uv, rings, |_| self.u_weight, self.v_period);
                    let gap = ((near[0] - uv[0]) * self.u_weight).hypot(near[1] - uv[1]);
                    grazed |= gap < tolerance;
                }
                if !chart_contains(uv, rings, self.v_period) {
                    continue;
                }
            }
            hits.push(t);
        }
        hits.sort_by(Scalar::total_cmp);
        // A ray through a shared grid edge registers on both its triangles.
        let merge = 1.0e-7 * self.diagonal / dot(d, d).sqrt().max(1.0e-30);
        hits.dedup_by(|a, b| (*a - *b).abs() <= merge);
        (hits, grazed)
    }
}

fn clamp_or_wrap(t: Scalar, range: [Scalar; 2], period: Option<Scalar>) -> Scalar {
    match period {
        Some(_) => {
            let span = range[1] - range[0];
            range[0] + (t - range[0]).rem_euclid(span)
        }
        None => t.clamp(range[0], range[1]),
    }
}

impl Brep {
    pub(super) fn sampled_patch(
        &self,
        surface: &BSplineSurface,
        face: &Face,
    ) -> Result<Sampled, &'static str> {
        if surface.control_points.first().is_none_or(Vec::is_empty) {
            return Err("B-spline surface has no control points");
        }
        let u_knots = expand(&surface.u_knots, &surface.u_multiplicities);
        let v_knots = expand(&surface.v_knots, &surface.v_multiplicities);
        let (rows, columns) = (
            surface.control_points.len(),
            surface.control_points[0].len(),
        );
        if u_knots.len() <= rows + surface.u_degree || v_knots.len() <= columns + surface.v_degree {
            return Err("B-spline surface knot vector is too short");
        }
        let control: Vec<Vec<[Scalar; D + 1]>> = surface
            .control_points
            .iter()
            .enumerate()
            .map(|(iu, row)| {
                row.iter()
                    .enumerate()
                    .map(|(iv, point)| {
                        let w = surface.weights.as_ref().map_or(1.0, |grid| grid[iu][iv]);
                        let mut homogeneous = [0.0; D + 1];
                        for k in 0..D {
                            homogeneous[k] = point[k].value() * w;
                        }
                        homogeneous[D] = w;
                        homogeneous
                    })
                    .collect()
            })
            .collect();
        let u_degree = surface.u_degree.min(rows.saturating_sub(1));
        let v_degree = surface.v_degree.min(columns.saturating_sub(1));
        let u_range = [u_knots[u_degree], u_knots[rows]];
        let v_range = [v_knots[v_degree], v_knots[columns]];
        let field = Field::BSpline {
            u_degree,
            v_degree,
            u_knots,
            v_knots,
            control,
        };
        // A grid dense enough to resolve the control net's own detail.
        let nu = (rows * 6).clamp(24, 48);
        let nv = (columns * 6).clamp(24, 48);
        self.assemble_sampled(field, u_range, v_range, nu, nv, face)
    }

    /// A surface of revolution as a sampled patch: `u` the revolution angle
    /// (`[0, TAU]`), `v` the normalized parameter along the profile polyline.
    pub(super) fn revolution_patch(
        &self,
        surface: &Revolution,
        face: &Face,
    ) -> Result<Sampled, &'static str> {
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let profile = self.revolution_profile(surface, face)?;
        let field = Field::Revolution {
            profile,
            origin,
            axis,
        };
        let profile_samples = match &field {
            Field::Revolution { profile, .. } => profile.len(),
            _ => unreachable!(),
        };
        self.assemble_sampled(
            field,
            [0.0, TAU],
            [0.0, 1.0],
            64,
            profile_samples.clamp(24, 48),
            face,
        )
    }

    /// Builds the [`Sampled`] shared machinery — grid, triangulation,
    /// periodicity, trim rings — around an already-constructed [`Field`].
    fn assemble_sampled(
        &self,
        field: Field,
        u_range: [Scalar; 2],
        v_range: [Scalar; 2],
        nu: usize,
        nv: usize,
        face: &Face,
    ) -> Result<Sampled, &'static str> {
        let u_of =
            |iu: usize| u_range[0] + (u_range[1] - u_range[0]) * iu as Scalar / (nu - 1) as Scalar;
        let v_of =
            |iv: usize| v_range[0] + (v_range[1] - v_range[0]) * iv as Scalar / (nv - 1) as Scalar;
        let grid: Vec<[Scalar; D]> = (0..nu)
            .flat_map(|iu| (0..nv).map(move |iv| (iu, iv)))
            .map(|(iu, iv)| field.point(u_of(iu), v_of(iv)))
            .collect();

        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for point in &grid {
            for k in 0..D {
                low[k] = low[k].min(point[k]);
                high[k] = high[k].max(point[k]);
            }
        }
        let diagonal = (0..D)
            .map(|k| (high[k] - low[k]).powi(2))
            .sum::<Scalar>()
            .sqrt();
        let seam = 1.0e-6 * diagonal.max(1.0e-12);
        let closed = |a: [Scalar; D], b: [Scalar; D]| {
            (0..D).map(|k| (a[k] - b[k]).powi(2)).sum::<Scalar>().sqrt() < seam
        };
        let u_period = (0..nv)
            .all(|iv| closed(grid[iv], grid[(nu - 1) * nv + iv]))
            .then_some(u_range[1] - u_range[0]);
        let v_period = (0..nu)
            .all(|iu| closed(grid[iu * nv], grid[iu * nv + nv - 1]))
            .then_some(v_range[1] - v_range[0]);

        let mut triangles = Vec::with_capacity((nu - 1) * (nv - 1) * 2);
        for iu in 0..nu - 1 {
            for iv in 0..nv - 1 {
                let (a, b, c, d) = (
                    iu * nv + iv,
                    (iu + 1) * nv + iv,
                    (iu + 1) * nv + iv + 1,
                    iu * nv + iv + 1,
                );
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }

        let mut u_weight = 0.0;
        let mut samples = 0usize;
        for iu in 0..4 {
            for iv in 0..4 {
                let u = u_range[0] + (u_range[1] - u_range[0]) * (iu as Scalar + 0.5) / 4.0;
                let v = v_range[0] + (v_range[1] - v_range[0]) * (iv as Scalar + 0.5) / 4.0;
                let h = (u_range[1] - u_range[0]) * 1.0e-4;
                u_weight += norm(diff(field.point(u + h, v), field.point(u - h, v), 2.0 * h));
                samples += 1;
            }
        }
        u_weight = (u_weight / samples as Scalar).max(1.0e-9);

        let mut sampled = Sampled {
            field,
            nu,
            nv,
            u_range,
            v_range,
            u_period,
            v_period,
            grid,
            triangles,
            rings: None,
            u_weight,
            sign: super::orientation(face.forward),
            tolerance: self.trim_tolerance(face, |curve| matches!(curve, Curve::Line(_))),
            diagonal,
            low,
            high,
        };
        sampled.rings = if sweeps_whole_surface(face) {
            None
        } else {
            let mut rings = Vec::new();
            for bound in &face.bounds {
                rings.push(self.sampled_ring(bound, &sampled)?);
            }
            Some(rings)
        };
        Ok(sampled)
    }

    /// The revolution's profile as a dense 3D polyline. A B-spline profile is
    /// sampled over its own span; a straight profile (a cone or cylinder some
    /// exporters emit as a revolution) is bounded by projecting the face's
    /// vertices onto it. A conic profile is not handled yet.
    fn revolution_profile(
        &self,
        surface: &Revolution,
        face: &Face,
    ) -> Result<Vec<[Scalar; D]>, &'static str> {
        const N: usize = 96;
        let raw = |point: &Coordinate<D>| -> [Scalar; D] { from_fn(|k| point[k].value()) };
        match &surface.curve {
            Curve::BSpline(bspline) => Ok(bspline.polyline(N).iter().map(raw).collect()),
            Curve::Line(line) => {
                let origin: [Scalar; D] = from_fn(|k| line.origin[k].value());
                let direction: [Scalar; D] = from_fn(|k| line.direction[k].value());
                let (mut lo, mut hi) = (Scalar::INFINITY, Scalar::NEG_INFINITY);
                for bound in &face.bounds {
                    for vertex in bound
                        .vertices(&self.edges)
                        .map_err(|_| "revolution profile: malformed bound")?
                    {
                        let rel: [Scalar; D] =
                            from_fn(|k| self.vertices[vertex][k].value() - origin[k]);
                        let t = dot(rel, direction);
                        lo = lo.min(t);
                        hi = hi.max(t);
                    }
                }
                if hi - lo <= 1.0e-12 {
                    return Err("revolution profile: straight profile has no extent here");
                }
                Ok((0..N)
                    .map(|i| {
                        let t = lo + (hi - lo) * i as Scalar / (N - 1) as Scalar;
                        from_fn(|k| origin[k] + t * direction[k])
                    })
                    .collect())
            }
            Curve::Circle(_) | Curve::Ellipse(_) => {
                Err("revolution of a conic profile is not yet meshable")
            }
        }
    }

    /// One bound of a B-spline face as a `(u, v)` polygon: every edge chorded
    /// and each chord vertex mapped through the surface's point inversion, `u`
    /// and `v` unwrapped against the running cursor where the surface is
    /// periodic.
    fn sampled_ring(&self, bound: &Loop, sampled: &Sampled) -> Result<Ring, &'static str> {
        let mut ring: Ring = Vec::new();
        let mut cursor: Option<[Scalar; 2]> = None;
        for half_edge in &bound.half_edges {
            let edge = self
                .edges
                .get(half_edge.edge)
                .ok_or("half-edge references a missing edge")?;
            let (start, end) = if half_edge.forward {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            let samples = self.edge_polyline(edge, start, end, half_edge.forward);
            let raw = |point: &Coordinate<D>| sampled.invert(from_fn(|k| point[k].value()));
            let mut point = cursor.unwrap_or_else(|| raw(&samples[0]));
            for sample in samples.iter().skip(1) {
                let next = raw(sample);
                let du = match sampled.u_period {
                    Some(period) => wrap_by(next[0] - point[0], period),
                    None => next[0] - point[0],
                };
                let dv = match sampled.v_period {
                    Some(period) => wrap_by(next[1] - point[1], period),
                    None => next[1] - point[1],
                };
                ring.push((point, None));
                point = [point[0] + du, point[1] + dv];
            }
            cursor = Some(point);
        }
        let (Some(last), Some(&(first, _))) = (cursor, ring.first()) else {
            return Err("trim ring on a B-spline face has no edges");
        };
        let u_span = sampled
            .u_period
            .unwrap_or(sampled.u_range[1] - sampled.u_range[0])
            .abs()
            .max(1.0e-12);
        let v_span = sampled
            .v_period
            .unwrap_or(sampled.v_range[1] - sampled.v_range[0])
            .abs()
            .max(1.0e-12);
        if (last[0] - first[0]).abs() > 1.0e-3 * u_span
            || (last[1] - first[1]).abs() > 1.0e-3 * v_span
        {
            return Err("unsupported trim ring on a B-spline face");
        }
        Ok(ring)
    }
}

/// The knots expanded to one entry per multiplicity.
fn expand(knots: &[f64], multiplicities: &[usize]) -> Vec<Scalar> {
    knots
        .iter()
        .zip(multiplicities)
        .flat_map(|(&knot, &multiplicity)| std::iter::repeat_n(knot, multiplicity))
        .collect()
}

fn diff(a: [Scalar; D], b: [Scalar; D], h: Scalar) -> [Scalar; D] {
    from_fn(|k| (a[k] - b[k]) / h)
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn cross(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [Scalar; D]) -> Scalar {
    dot(v, v).sqrt()
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let n = norm(v);
    (n > 1.0e-30).then(|| v.map(|x| x / n))
}

fn blend(a: [Scalar; 2], b: [Scalar; 2], c: [Scalar; 2], [wa, wb, wc]: [Scalar; 3]) -> [Scalar; 2] {
    [
        wa * a[0] + wb * b[0] + wc * c[0],
        wa * a[1] + wb * b[1] + wc * c[1],
    ]
}

/// Möller-Trumbore: `(t, [wa, wb, wc])` where the ray `o + t*d` meets triangle
/// `a b c`, barycentric weights summing to one. `d` need not be unit.
fn intersect(
    o: [Scalar; D],
    d: [Scalar; D],
    a: [Scalar; D],
    b: [Scalar; D],
    c: [Scalar; D],
) -> Option<(Scalar, [Scalar; 3])> {
    let ab: [Scalar; D] = from_fn(|k| b[k] - a[k]);
    let ac: [Scalar; D] = from_fn(|k| c[k] - a[k]);
    let p = cross(d, ac);
    let determinant = dot(ab, p);
    if determinant.abs() < 1.0e-18 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let tvec: [Scalar; D] = from_fn(|k| o[k] - a[k]);
    let wb = dot(tvec, p) * inverse;
    if !(-1.0e-9..=1.0 + 1.0e-9).contains(&wb) {
        return None;
    }
    let q = cross(tvec, ab);
    let wc = dot(d, q) * inverse;
    if wc < -1.0e-9 || wb + wc > 1.0 + 1.0e-9 {
        return None;
    }
    let t = dot(ac, q) * inverse;
    Some((t, [1.0 - wb - wc, wb, wc]))
}
