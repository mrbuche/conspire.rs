use super::YieldSurface;
use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, SquareMatrix, TensorArray, TensorRank2, Vector},
    mechanics::{FlowDirectionPlastic, MandelStressElastic, Scalar, StretchingRatePlastic},
    units::{Dissipation, Rate, Stress},
};
use std::array::from_fn;

type Matrix3 = [[Scalar; 3]; 3];

/// The Hill yield surface.
///
/// The equivalent stress is the orthotropic quadratic form
/// ```math
/// \phi^2 = F(a_{22}-a_{33})^2 + G(a_{33}-a_{11})^2 + H(a_{11}-a_{22})^2
///        + 2L\,a_{23}^2 + 2M\,a_{31}^2 + 2N\,a_{12}^2,
/// \qquad \mathbf{a} = \mathrm{sym}(\mathbf{M}_\mathrm{e}'),
/// ```
/// in the axes of the intermediate configuration. It is normalized so that
/// $`F=G=H=1/3`$ and $`L=M=N=1`$ gives $`\phi=|\mathbf{M}_\mathrm{e}'|`$, the
/// [`VonMises`](super::VonMises) surface, and the coefficients must be positive.
#[derive(Clone, Debug)]
pub struct Hill {
    /// The coefficient $`F`$.
    pub f: Scalar,
    /// The coefficient $`G`$.
    pub g: Scalar,
    /// The coefficient $`H`$.
    pub h: Scalar,
    /// The coefficient $`L`$.
    pub l: Scalar,
    /// The coefficient $`M`$.
    pub m: Scalar,
    /// The coefficient $`N`$.
    pub n: Scalar,
}

fn symmetric_part<I, J, U>(tensor: &TensorRank2<3, I, J, U>) -> Matrix3 {
    from_fn(|i| from_fn(|j| 0.5 * (tensor[i][j].value() + tensor[j][i].value())))
}

fn contract(a: &Matrix3, b: &Matrix3) -> Scalar {
    (0..3)
        .map(|i| (0..3).map(|j| a[i][j] * b[i][j]).sum::<Scalar>())
        .sum()
}

impl Hill {
    /// The quadratic form as a symmetric operator, $`\mathcal{H}:\mathbf{a}`$.
    fn operator(&self, a: &Matrix3) -> Matrix3 {
        let Self { f, g, h, l, m, n } = self;
        [
            [
                g * (a[0][0] - a[2][2]) + h * (a[0][0] - a[1][1]),
                n * a[0][1],
                m * a[0][2],
            ],
            [
                n * a[0][1],
                f * (a[1][1] - a[2][2]) + h * (a[1][1] - a[0][0]),
                l * a[1][2],
            ],
            [
                m * a[0][2],
                l * a[1][2],
                f * (a[2][2] - a[1][1]) + g * (a[2][2] - a[0][0]),
            ],
        ]
    }
    fn equivalent(&self, a: &Matrix3) -> Scalar {
        contract(a, &self.operator(a)).max(0.0).sqrt()
    }
}

impl YieldSurface for Hill {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(Stress::pascals(
            self.equivalent(&symmetric_part(deviatoric_mandel_stress)),
        ))
    }
    /// ```math
    /// \mathbf{N} = \frac{\mathcal{H}:\mathbf{a}}{\phi}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let a = symmetric_part(deviatoric_mandel_stress);
        let magnitude = self.equivalent(&a);
        if magnitude == 0.0 {
            return Ok(FlowDirectionPlastic::zero());
        }
        let operator = self.operator(&a);
        Ok(FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| {
            from_fn::<_, 3, _>(|j| operator[i][j] / magnitude)
        })))
    }
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{a}
    /// = \frac{1}{\phi}\left(\mathcal{H}:\mathrm{d}\mathbf{a} - \mathbf{N}\,(\mathbf{N}:\mathrm{d}\mathbf{a})\right)
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let a = symmetric_part(deviatoric_mandel_stress);
        let magnitude = self.equivalent(&a);
        if magnitude == 0.0 {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction: Matrix3 = {
            let operator = self.operator(&a);
            from_fn(|i| from_fn(|j| operator[i][j] / magnitude))
        };
        let d_a = symmetric_part(increment);
        let d_operator = self.operator(&d_a);
        let slope = contract(&direction, &d_a);
        Ok(FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| {
            from_fn::<_, 3, _>(|j| (d_operator[i][j] - direction[i][j] * slope) / magnitude)
        })))
    }
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\sqrt{\mathbf{D}_\mathrm{p}:\mathcal{H}^+:\mathbf{D}_\mathrm{p}}
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let Self { f, g, h, l, m, n } = *self;
        let d = symmetric_part(&plastic_stretching_rate);
        // the normal block of the form annihilates the hydrostatic direction, so adding
        // its outer product makes it invertible without changing it on the deviatoric one
        let normal = SquareMatrix::from([
            [g + h + 1.0, 1.0 - h, 1.0 - g],
            [1.0 - h, f + h + 1.0, 1.0 - f],
            [1.0 - g, 1.0 - f, f + g + 1.0],
        ]);
        let rates = Vector::from(vec![d[0][0], d[1][1], d[2][2]]);
        let solved = normal
            .solve_lu(&rates)
            .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), self))?;
        let squared = (0..3).map(|i| rates[i] * solved[i]).sum::<Scalar>()
            + 2.0 * (d[1][2] * d[1][2] / l + d[0][2] * d[0][2] / m + d[0][1] * d[0][1] / n);
        Ok(yield_stress * Quantity::<Rate>::new(squared.max(0.0).sqrt()))
    }
}
