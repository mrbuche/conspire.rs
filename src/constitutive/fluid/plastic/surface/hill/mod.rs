use super::YieldSurface;
use crate::{
    constitutive::ConstitutiveError,
    math::{ContractWith, Erase, Quantity, SquareMatrix, Tensor, TensorArray, TensorRank2, Vector},
    mechanics::{FlowDirectionPlastic, MandelStressElastic, Scalar, StretchingRatePlastic},
    units::{Dissipation, Rate, Stress},
};

#[doc = include_str!("doc.md")]
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

impl Hill {
    /// The quadratic form as a symmetric operator, $`\mathcal{H}:\mathbf{a}`$.
    fn operator<I, U>(&self, a: &TensorRank2<3, I, I, U>) -> TensorRank2<3, I, I, U> {
        let Self { f, g, h, l, m, n } = *self;
        let (a_11, a_22, a_33) = (a[0][0], a[1][1], a[2][2]);
        TensorRank2::from([
            [
                (a_11 - a_33) * g + (a_11 - a_22) * h,
                a[0][1] * n,
                a[0][2] * m,
            ],
            [
                a[0][1] * n,
                (a_22 - a_33) * f + (a_22 - a_11) * h,
                a[1][2] * l,
            ],
            [
                a[0][2] * m,
                a[1][2] * l,
                (a_33 - a_22) * f + (a_33 - a_11) * g,
            ],
        ])
    }
    fn equivalent(&self, a: &MandelStressElastic) -> Quantity<Stress> {
        let operator = self.operator(a);
        let squared = a.erase().contract_with(operator.erase()).value();
        Stress::pascals(squared.max(0.0).sqrt())
    }
}

impl YieldSurface for Hill {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.equivalent(&deviatoric_mandel_stress.symmetric_part()))
    }
    /// ```math
    /// \mathbf{N} = \frac{\mathcal{H}:\mathbf{a}}{\phi}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let a = deviatoric_mandel_stress.symmetric_part();
        let magnitude = self.equivalent(&a);
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(self.operator(&a) / magnitude)
        }
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
        let a = deviatoric_mandel_stress.symmetric_part();
        let magnitude = self.equivalent(&a);
        if magnitude.is_zero() {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction = self.operator(&a) / magnitude;
        let d_a = increment.symmetric_part();
        let slope = d_a.contract_with(&direction);
        Ok((self.operator(&d_a) - &(direction * slope)) / magnitude)
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
        let d = plastic_stretching_rate.symmetric_part();
        let normal = SquareMatrix::from([
            [g + h + 1.0, 1.0 - h, 1.0 - g],
            [1.0 - h, f + h + 1.0, 1.0 - f],
            [1.0 - g, 1.0 - f, f + g + 1.0],
        ]);
        let rates = Vector::from(vec![d[0][0].value(), d[1][1].value(), d[2][2].value()]);
        let solved = normal
            .solve_lu(&rates)
            .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), self))?;
        let (d_12, d_13, d_23) = (d[0][1].value(), d[0][2].value(), d[1][2].value());
        let squared = (0..3).map(|i| rates[i] * solved[i]).sum::<Scalar>()
            + 2.0 * (d_23 * d_23 / l + d_13 * d_13 / m + d_12 * d_12 / n);
        Ok(yield_stress * Quantity::<Rate>::new(squared.max(0.0).sqrt()))
    }
}
