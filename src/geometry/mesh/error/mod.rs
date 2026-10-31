use crate::math::{Scalar, Style, StyledError, styled_error};

/// Possible errors for meshes.
pub enum MeshError {
    Budget { theta: Scalar, sweeps: usize },
    HexahedraRequired,
    Minimization,
    Stall { theta: Scalar, sweeps: usize },
    SurfaceFit,
}

impl StyledError for MeshError {
    fn message(&self, style: &Style) -> String {
        let h = style.headline;
        match self {
            Self::Budget { theta, sweeps } => format!(
                "{h}The scaled Jacobian pass used all {sweeps} sweeps without reaching {theta:.2}."
            ),
            Self::HexahedraRequired => format!("{h}The mesh must consist of hexahedra."),
            Self::Minimization => format!("{h}Could not minimize the energy."),
            Self::Stall { theta, sweeps } => format!(
                "{h}The scaled Jacobian pass stalled after {sweeps} sweeps short of {theta:.2}."
            ),
            Self::SurfaceFit => format!("{h}Could not fit the surface."),
        }
    }
}

styled_error!(MeshError);
