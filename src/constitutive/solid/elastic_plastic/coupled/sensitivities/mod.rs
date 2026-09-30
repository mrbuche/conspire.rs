use super::{
    ElasticPlastic, SIZE, Unknowns, basis, failure, increment, linearization::Linearization,
};
use crate::{
    constitutive::ConstitutiveError,
    math::{
        ContractSecondWithFirst, ContractThirdFourthWithFirstSecond, ContractWith, Intermediate,
        Quantity, Rank2, Reference, Tensor, TensorArray, TensorRank4,
    },
    mechanics::{
        DeformationGradient, DeformationGradientPlastic, FlowDirectionPlastic, MandelStressElastic,
        Scalar,
    },
    units::Stress,
};

/// Everything evaluated once per iterate: the plastic deformation gradient, the
/// deviatoric Mandel stress with its equivalent stress and flow direction, the residual,
/// and the multiplier with the hardening modulus at its plastic strain, which the
/// Jacobian needs.
pub(super) struct Iterate {
    pub(super) plastic: DeformationGradientPlastic,
    deviatoric: MandelStressElastic,
    unit: FlowDirectionPlastic,
    direction: FlowDirectionPlastic,
    magnitude: Quantity<Stress>,
    pub(super) residual: Unknowns,
    gamma: Scalar,
    hardening_modulus: Scalar,
}

impl Iterate {
    pub(super) fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        strain_n: Scalar,
        x: &Unknowns,
    ) -> Result<Self, ConstitutiveError> {
        let plastic = increment(x)
            .expm()
            .map_err(|error| failure(model, &error))?
            * f_p_n;
        let deviatoric: MandelStressElastic = model.mandel_stress(f, &plastic)?.deviatoric();
        let magnitude = model.equivalent_stress(&deviatoric)?;
        let unit = model.flow_direction(&deviatoric)?;
        let direction = unit.symmetric_part();
        let mut residual = [0.0; SIZE];
        let flow = increment(x) - &direction * x[9];
        (0..3).for_each(|i| (0..3).for_each(|j| residual[3 * i + j] = flow[i][j].value()));
        residual[9] = model
            .yield_function(&deviatoric, Quantity::new(strain_n + x[9]))?
            .value();
        Ok(Self {
            plastic,
            deviatoric,
            unit,
            direction,
            magnitude,
            residual,
            gamma: x[9],
            hardening_modulus: model
                .hardening_modulus(Quantity::new(strain_n + x[9]))?
                .value(),
        })
    }
}

/// The linearization at an iterate together with the slopes
/// $`\partial\mathbf{F}_\mathrm{p}/\partial E_{ab}`$ of the exponential map.
pub(super) struct Sensitivities<'a, C> {
    model: &'a C,
    pub(super) linearization: Linearization,
    iterate: &'a Iterate,
    slopes: TensorRank4<3, Intermediate, Reference, Intermediate, Intermediate>,
}

impl<'a, C: ElasticPlastic> Sensitivities<'a, C> {
    pub(super) fn new(
        model: &'a C,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        x: &Unknowns,
        iterate: &'a Iterate,
    ) -> Result<Self, ConstitutiveError> {
        let slopes = increment(x)
            .dexpm()
            .map_err(|error| failure(model, &error))?
            .contract_second_with_first(f_p_n);
        Ok(Self {
            model,
            linearization: Linearization::new(model, f, &iterate.plastic)?,
            iterate,
            slopes,
        })
    }

    /// The slope of the plastic deformation gradient along the direction $`E_{ab}`$.
    pub(super) fn slope(&self, a: usize, b: usize) -> DeformationGradientPlastic {
        (&self.slopes).contract_third_fourth_with_first_second(&basis(a, b))
    }

    /// The slope of the symmetrized flow direction and of the equivalent stress along a
    /// Mandel stress increment.
    pub(super) fn direction_slope(
        &self,
        d_m: &MandelStressElastic,
    ) -> Result<(FlowDirectionPlastic, Scalar), ConstitutiveError> {
        let Iterate {
            unit,
            magnitude,
            deviatoric,
            ..
        } = self.iterate;
        // the equivalent stress is not differentiable where the deviator vanishes, and
        // the flow direction is zero there: the trial state is elastic
        if magnitude.is_zero() {
            return Ok((FlowDirectionPlastic::zero(), 0.0));
        }
        let increment = d_m.deviatoric();
        let d_magnitude = increment.contract_with(unit).value();
        let d_unit = self.model.flow_direction_slope(deviatoric, &increment)?;
        Ok((d_unit.symmetric_part(), d_magnitude))
    }

    /// The Jacobian of the residual with respect to $`(\mathbf{E},\Delta\gamma)`$.
    pub(super) fn jacobian(&self) -> Result<[[Scalar; SIZE]; SIZE], ConstitutiveError> {
        let Iterate {
            gamma,
            hardening_modulus,
            ..
        } = self.iterate;
        let mut jacobian = [[0.0; SIZE]; SIZE];
        for a in 0..3 {
            for b in 0..3 {
                let d_m = self
                    .linearization
                    .mandel_derivative(&DeformationGradient::zero(), &self.slope(a, b));
                let (d_direction, d_magnitude) = self.direction_slope(&d_m)?;
                for i in 0..3 {
                    for j in 0..3 {
                        let identity = if i == a && j == b { 1.0 } else { 0.0 };
                        jacobian[3 * i + j][3 * a + b] =
                            identity - gamma * d_direction[i][j].value();
                    }
                }
                jacobian[9][3 * a + b] = d_magnitude;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                jacobian[3 * i + j][9] = -self.iterate.direction[i][j].value();
            }
        }
        jacobian[9][9] = -*hardening_modulus;
        Ok(jacobian)
    }
}
