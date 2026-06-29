mod adaptive;
mod triangles;
mod uniform;

use crate::{geometry::mesh::Mesh, math::Scalar};

const D: usize = 3;

/// A remeshing scheme: a number of iterations and a metric kind with its sizing.
pub struct Remeshing {
    /// Number of remeshing iterations.
    pub iterations: usize,
    /// The metric kind (isotropic or anisotropic) and its sizing.
    pub kind: RemeshingKind,
}

/// The remeshing metric: isotropic (scalar size) or anisotropic (tensor metric).
pub enum RemeshingKind {
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
        let Remeshing { iterations, kind } = remeshing;
        match kind {
            RemeshingKind::Isotropic(sizing) => match sizing {
                IsotropicSizing::Uniform { length } => self.uniform_remesh(iterations, length),
                IsotropicSizing::Adaptive {
                    tolerance,
                    minimum,
                    maximum,
                    gradation,
                } => self.adaptive_remesh(iterations, tolerance, minimum, maximum, gradation),
            },
            RemeshingKind::Anisotropic(_) => Err("anisotropic remeshing is not implemented yet"),
        }
    }
}
