#[cfg(test)]
mod test;

use crate::math::Scalar;
use std::mem::swap;

const ARMIJO: Scalar = 1.0e-4;
const CURVATURE_FLOOR: Scalar = 1.0e-12;

pub(crate) trait Objective {
    fn value(&mut self, x: &[Scalar]) -> Scalar;
    fn gradient(&mut self, x: &[Scalar], gradient: &mut [Scalar]);
}

pub(crate) struct Settings {
    pub backtracks: usize,
    pub convergence: Scalar,
    pub history: usize,
    pub iterations: usize,
}

pub(crate) struct Outcome {
    pub initial: Scalar,
    pub settled: bool,
    pub value: Scalar,
}

pub(crate) fn minimize(
    objective: &mut impl Objective,
    x: &mut [Scalar],
    typical: Scalar,
    settings: &Settings,
) -> Outcome {
    let mut history: Vec<(Vec<Scalar>, Vec<Scalar>)> = Vec::new();
    let mut gradient = vec![0.0; x.len()];
    let mut updated = vec![0.0; x.len()];
    let mut trial = x.to_vec();
    objective.gradient(x, &mut gradient);
    let mut value = objective.value(x);
    let initial = value;
    let mut settled = false;
    for iteration in 0..settings.iterations {
        let magnitude = norm(&gradient);
        if magnitude / norm(x).max(1.0) < settings.convergence {
            settled = iteration == 0;
            break;
        }
        let d = direction(&gradient, &history, typical / magnitude);
        let slope = dot(&gradient, &d);
        if slope >= 0.0 {
            history.clear();
            continue;
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..settings.backtracks {
            trial
                .iter_mut()
                .zip(x.iter().zip(&d))
                .for_each(|(t, (x, d))| *t = x + d * step);
            let candidate = objective.value(&trial);
            if candidate <= value + ARMIJO * step * slope {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        let Some(candidate) = accepted else {
            if history.is_empty() {
                break;
            }
            history.clear();
            continue;
        };
        let s: Vec<Scalar> = d.iter().map(|d| d * step).collect();
        x.iter_mut().zip(&s).for_each(|(x, s)| *x += s);
        updated.iter_mut().for_each(|g| *g = 0.0);
        objective.gradient(x, &mut updated);
        let y: Vec<Scalar> = updated.iter().zip(&gradient).map(|(n, o)| n - o).collect();
        if dot(&s, &y) > CURVATURE_FLOOR * norm(&s) * norm(&y) {
            if history.len() == settings.history {
                history.remove(0);
            }
            history.push((s, y));
        }
        swap(&mut gradient, &mut updated);
        value = candidate;
    }
    Outcome {
        initial,
        settled,
        value,
    }
}

fn direction(
    gradient: &[Scalar],
    history: &[(Vec<Scalar>, Vec<Scalar>)],
    fallback: Scalar,
) -> Vec<Scalar> {
    let mut q = gradient.to_vec();
    let mut alphas = vec![0.0; history.len()];
    let rhos: Vec<Scalar> = history.iter().map(|(s, y)| 1.0 / dot(y, s)).collect();
    history.iter().enumerate().rev().for_each(|(k, (s, y))| {
        alphas[k] = rhos[k] * dot(s, &q);
        q.iter_mut().zip(y).for_each(|(q, y)| *q -= alphas[k] * y)
    });
    let gamma = history
        .last()
        .map_or(fallback, |(s, y)| dot(s, y) / dot(y, y));
    q.iter_mut().for_each(|q| *q *= gamma);
    history.iter().enumerate().for_each(|(k, (s, y))| {
        let beta = rhos[k] * dot(y, &q);
        q.iter_mut()
            .zip(s)
            .for_each(|(q, s)| *q += (alphas[k] - beta) * s)
    });
    q.iter_mut().for_each(|q| *q = -*q);
    q
}

fn dot(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.algebraic_mul(*b))
        .fold(0.0, Scalar::algebraic_add)
}

fn norm(a: &[Scalar]) -> Scalar {
    dot(a, a).sqrt()
}
