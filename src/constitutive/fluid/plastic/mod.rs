//! Plastic fluid constitutive models.

#[cfg(test)]
mod test;

use crate::{
    constitutive::ConstitutiveError,
    math::{
        Quantity, SquareMatrix, Tensor, TensorArray, TensorRank2, TensorTuple, TensorTupleVec,
        Vector,
    },
    mechanics::{
        DeformationGradientPlastic, FlowDirectionPlastic, MandelStressElastic, Scalar,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress},
};
use std::{array::from_fn, fmt::Debug};

/// Rate-independent plastic state variables $`(\mathbf{F}_\mathrm{p},\,\varepsilon_\mathrm{p})`$.
pub type PlasticStateVariables = TensorTuple<DeformationGradientPlastic, Quantity>;

/// The history of the rate-independent plastic state variables.
pub type PlasticStateVariablesHistory = TensorTupleVec<DeformationGradientPlastic, Quantity>;

/// Required methods for plastic fluid constitutive models.
pub trait Plastic
where
    Self: Clone + Debug,
{
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Quantity<Stress>;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Quantity<Stress>;
    /// Calculates and returns the yield stress.
    ///
    /// ```math
    /// Y = Y_0 + H\,\varepsilon_\mathrm{p}
    /// ```
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.initial_yield_stress() + self.hardening_slope() * equivalent_plastic_strain)
    }
    /// Calculates and returns the hardening modulus, the derivative of the yield stress
    /// with respect to the equivalent plastic strain.
    ///
    /// ```math
    /// \frac{\mathrm{d}Y}{\mathrm{d}\varepsilon_\mathrm{p}} = H
    /// ```
    ///
    /// This is the derivative of [`Self::yield_stress`]: a model that overrides one
    /// must override the other, and a wrapper must forward both.
    fn hardening_modulus(
        &self,
        _equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.hardening_slope())
    }
}

/// Required methods for rate-independent (yield-surface) plastic fluid constitutive models.
///
/// The yield surface is $`\phi(\mathbf{M}_\mathrm{e}') = Y(\varepsilon_\mathrm{p})`$ with
/// the equivalent stress $`\phi`$ positively homogeneous of degree one, smooth away from
/// the origin, and dependent on the deviatoric Mandel stress only, and the flow is
/// associative. The defaults are von Mises, $`\phi = |\mathbf{M}_\mathrm{e}'|`$; another
/// surface overrides [`Self::equivalent_stress`], [`Self::flow_direction`],
/// [`Self::flow_direction_slope`] and [`Self::dissipation_potential`], and a wrapper must
/// forward all of them.
pub trait RateIndependentPlastic
where
    Self: Plastic,
{
    /// Returns the initial state of the variables.
    fn initial_state(&self) -> PlasticStateVariables {
        (DeformationGradientPlastic::identity(), Quantity::default()).into()
    }
    /// Calculates and returns the equivalent stress.
    ///
    /// ```math
    /// \phi(\mathbf{M}_\mathrm{e}') = |\mathbf{M}_\mathrm{e}'|
    /// ```
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(deviatoric_mandel_stress.norm())
    }
    /// Calculates and returns the yield function.
    ///
    /// ```math
    /// f(\mathbf{M}_\mathrm{e}',\varepsilon_\mathrm{p}) = \phi(\mathbf{M}_\mathrm{e}') - Y(\varepsilon_\mathrm{p})
    /// ```
    fn yield_function(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.equivalent_stress(deviatoric_mandel_stress)?
            - self.yield_stress(equivalent_plastic_strain)?)
    }
    /// Calculates and returns the associative plastic flow direction.
    ///
    /// ```math
    /// \mathbf{N} = \frac{\partial\phi}{\partial\mathbf{M}_\mathrm{e}'} = \frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress / magnitude)
        }
    }
    /// Calculates and returns the derivative of the flow direction along a deviatoric
    /// Mandel stress increment.
    ///
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{M}_\mathrm{e}'
    /// = \frac{1}{|\mathbf{M}_\mathrm{e}'|}\left(\mathrm{d}\mathbf{M}_\mathrm{e}' - \mathbf{N}\,(\mathbf{N}:\mathrm{d}\mathbf{M}_\mathrm{e}')\right)
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm().value();
        if magnitude == 0.0 {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction: [[Scalar; 3]; 3] =
            from_fn(|i| from_fn(|j| deviatoric_mandel_stress[i][j].value() / magnitude));
        let slope: Scalar = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| direction[i][j] * increment[i][j].value())
                    .sum::<Scalar>()
            })
            .sum();
        Ok(FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| {
            from_fn::<_, 3, _>(|j| (increment[i][j].value() - direction[i][j] * slope) / magnitude)
        })))
    }
    /// Calculates and returns the plastic stretching rate.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = \dot{\gamma}\,\mathbf{N},\qquad
    /// \dot{\gamma}\geq 0,\quad f\leq 0,\quad \dot{\gamma}f = 0
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        plastic_multiplier: Quantity<Rate>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        Ok(self.flow_direction(deviatoric_mandel_stress)? * plastic_multiplier)
    }
    /// Calculates and returns the plastic dissipation potential.
    ///
    /// ```math
    /// \phi(\mathbf{D}_\mathrm{p}) = Y\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(yield_stress * plastic_stretching_rate.norm())
    }
}

/// The rate-independent von Mises plastic flow model.
#[derive(Clone, Debug)]
pub struct PlasticFlow {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
}

impl Plastic for PlasticFlow {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl RateIndependentPlastic for PlasticFlow {}

/// The rate-independent von Mises plastic flow model with Voce (saturating) isotropic
/// hardening.
///
/// ```math
/// Y(\varepsilon_\mathrm{p}) = Y_0 + H\,\varepsilon_\mathrm{p} + Q\left(1 - e^{-b\,\varepsilon_\mathrm{p}}\right)
/// ```
#[derive(Clone, Debug)]
pub struct VoceFlow {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The linear hardening slope $`H`$, which persists after saturation.
    pub hardening_slope: Quantity<Stress>,
    /// The saturation stress $`Q`$ added to the yield stress at full saturation.
    pub saturation_stress: Quantity<Stress>,
    /// The saturation rate $`b`$.
    pub saturation_rate: Scalar,
}

impl Plastic for VoceFlow {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    /// The initial hardening slope $`H + Qb`$, at zero plastic strain.
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope + self.saturation_stress * self.saturation_rate
    }
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let saturation = 1.0 - (-self.saturation_rate * equivalent_plastic_strain.value()).exp();
        Ok(self.yield_stress
            + self.hardening_slope * equivalent_plastic_strain
            + self.saturation_stress * saturation)
    }
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (-self.saturation_rate * equivalent_plastic_strain.value()).exp();
        Ok(self.hardening_slope + self.saturation_stress * (self.saturation_rate * decay))
    }
}

impl RateIndependentPlastic for VoceFlow {}

type Matrix3 = [[Scalar; 3]; 3];

/// The rate-independent Hill plastic flow model with linear isotropic hardening.
///
/// The equivalent stress is the orthotropic quadratic form
/// ```math
/// \phi^2 = F(a_{22}-a_{33})^2 + G(a_{33}-a_{11})^2 + H(a_{11}-a_{22})^2
///        + 2L\,a_{23}^2 + 2M\,a_{31}^2 + 2N\,a_{12}^2,
/// \qquad \mathbf{a} = \mathrm{sym}(\mathbf{M}_\mathrm{e}'),
/// ```
/// in the axes of the intermediate configuration. It is normalized so that
/// $`F=G=H=1/3`$ and $`L=M=N=1`$ gives $`\phi=|\mathbf{M}_\mathrm{e}'|`$, the
/// [`PlasticFlow`] model, and the coefficients must be positive.
#[derive(Clone, Debug)]
pub struct Hill {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
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
        let Self {
            f, g, h, l, m, n, ..
        } = self;
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

impl Plastic for Hill {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl RateIndependentPlastic for Hill {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(Stress::pascals(
            self.equivalent(&symmetric_part(deviatoric_mandel_stress)),
        ))
    }
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
    /// The dual gauge of the equivalent stress, so that $`\phi_\mathrm{d} = Y\dot\gamma`$
    /// for $`\mathbf{D}_\mathrm{p} = \dot\gamma\,\mathbf{N}`$,
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\sqrt{\mathbf{D}_\mathrm{p}:\mathcal{H}^+:\mathbf{D}_\mathrm{p}}.
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let Self {
            f, g, h, l, m, n, ..
        } = *self;
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
