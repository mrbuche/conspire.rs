//! Meshing B-reps.
//!
//! Temporary: routes through `Brep::tessellate` and the tessellation's
//! octree-dual background. The direct B-rep mesher will replace this.

#[cfg(test)]
mod test;

use super::brep::Brep;
use crate::{
    geometry::{
        mesh::{Class, Mesh},
        ntree::Balancing,
    },
    math::Scalar,
};

impl Brep {
    /// Hexahedral background for this solid: the dual of an octree fitted to
    /// its tessellation, every cell classified inside, cut, or outside.
    ///
    /// `balancing` must be `Strong(1)` or `Weak(1)`. Pair with
    /// [`Tessellation::cut`](crate::geometry::mesh::Tessellation::cut) for a
    /// fitted mesh.
    pub fn dual_background(
        &self,
        balancing: Balancing,
        scale: Scalar,
    ) -> Result<(Mesh<3>, Vec<Class>), &'static str> {
        self.tessellate()?.dual_background(balancing, scale)
    }
}
