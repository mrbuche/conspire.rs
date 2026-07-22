pub(crate) mod adaptive;
#[cfg(test)]
mod test;
mod triangles;
mod uniform;

use crate::{geometry::mesh::Mesh, math::Scalar};

const D: usize = 3;

/// A remeshing scheme with a number of iterations and a metric.
pub struct Remeshing {
    /// Number of remeshing iterations.
    pub iterations: usize,
    /// The metric (isotropic or anisotropic).
    pub metric: RemeshingMetric,
}

/// Different metrics for remeshing.
pub enum RemeshingMetric {
    /// Isotropic remeshing (circular/spherical target metric).
    Isotropic(IsotropicSizing),
    /// Anisotropic remeshing (directional, curvature-aligned target metric).
    Anisotropic(AnisotropicSizing),
}

/// Sizing for isotropic remeshing.
pub enum IsotropicSizing {
    /// Constant target edge length over the whole mesh ([`None`] = mean edge length).
    Uniform { length: Option<Scalar> },
    /// Curvature-driven scalar size field (Dunyach).
    Adaptive {
        tolerance: Scalar,
        minimum: Scalar,
        maximum: Scalar,
        gradation: Scalar,
    },
}

/// Sizing for anisotropic remeshing (not implemented yet; parameters to be determined).
pub enum AnisotropicSizing {
    Uniform,
    Adaptive,
}

impl Mesh<D> {
    pub fn remesh(self, remeshing: Remeshing) -> Result<Self, &'static str> {
        let Remeshing { iterations, metric } = remeshing;
        match metric {
            RemeshingMetric::Isotropic(sizing) => match sizing {
                IsotropicSizing::Uniform { length } => self.uniform_remesh(iterations, length),
                IsotropicSizing::Adaptive {
                    tolerance,
                    minimum,
                    maximum,
                    gradation,
                } => self.adaptive_remesh(iterations, tolerance, minimum, maximum, gradation),
            },
            RemeshingMetric::Anisotropic(_) => Err("anisotropic remeshing is not implemented yet"),
        }
    }
}
