#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar, TensorArray,
        special::{inverse_langevin, langevin, langevin_derivative},
    },
    mechanics::{CurrentCoordinate, CurrentCoordinates},
    physics::molecular::single_chain::{
        Ensemble, Inextensible, Isometric, Isotensional, Legendre, MonteCarlo, SingleChain,
        SingleChainError, Thermodynamics,
    },
};
use std::f64::consts::{PI, TAU};

/// The freely-jointed chain model.
#[derive(Clone, Debug)]
pub struct FreelyJointedChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for FreelyJointedChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for FreelyJointedChain {
    /// ```math
    /// \lim_{\eta\to\infty}\gamma(\eta) = 1
    /// ```
    fn maximum_nondimensional_extension(&self) -> Scalar {
        1.0
    }
}

impl Thermodynamics for FreelyJointedChain {
    fn ensemble(&self) -> Ensemble {
        self.ensemble
    }
}

impl Isometric for FreelyJointedChain {
    fn nondimensional_helmholtz_free_energy(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(0.0)
        } else {
            let [s0, _, _] = treloar_sums(self.number_of_links(), nondimensional_extension);
            Ok(nondimensional_extension.abs().ln() - s0.ln())
        }
    }
    /// ```math
    /// \eta(\gamma) = \frac{1}{N_b\gamma} + \left(\frac{1}{2} - \frac{1}{N_b}\right)\frac{\sum_{s=0}^{s_\mathrm{max}}(-1)^s\binom{N_b}{s}\left(m - \frac{s}{N_b}\right)^{N_b - 3}}{\sum_{s=0}^{s_\mathrm{max}}(-1)^s\binom{N_b}{s}\left(m - \frac{s}{N_b}\right)^{N_b - 2}}
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(0.0)
        } else {
            let [s0, s1, _] = treloar_sums(self.number_of_links(), nondimensional_extension);
            let n = self.number_of_links() as Scalar;
            Ok((1.0 / nondimensional_extension + (0.5 * n - 1.0) * s1 / s0) / n)
        }
    }
    /// ```math
    /// k(\gamma) = \frac{\partial\eta}{\partial\gamma}
    /// ```
    fn nondimensional_stiffness(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension == 0.0 {
            Ok(Scalar::NAN)
        } else {
            let [s0, s1, s2] = treloar_sums(self.number_of_links(), nondimensional_extension);

            if !s0.is_finite() || s0 == 0.0 {
                return Ok(Scalar::NAN);
            }

            let n = self.number_of_links() as Scalar;
            let p = n - 2.0;
            let b = (0.5 * n - 1.0) / n;

            let ds0dx = -(p / 2.0) * s1;
            let ds1dx = -((p - 1.0) / 2.0) * s2;
            let d_ratio_dx = (ds1dx * s0 - s1 * ds0dx) / (s0 * s0);

            Ok(-1.0 / (n * nondimensional_extension * nondimensional_extension) + b * d_ratio_dx)
        }
    }
    /// ```math
    /// \mathcal{P}(\gamma) = \frac{1}{8\pi\gamma}\frac{N_b^{N_b}}{(N_b - 2)!}\sum_{s=0}^{s_\mathrm{max}}(-1)^s\binom{N_b}{s}\left(m - \frac{s}{N_b}\right)^{N_b - 2}
    /// ```
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        if nondimensional_extension <= 0.0 || nondimensional_extension >= 1.0 {
            Ok(0.0)
        } else {
            let number_of_links = self.number_of_links();
            let [s0, _, _] = treloar_sums(number_of_links, nondimensional_extension);
            let n = number_of_links as Scalar;
            let factorial_n_minus_2 = (1..=(number_of_links - 2))
                .map(|i| i as Scalar)
                .product::<Scalar>();
            Ok((n.powf(n) / (8.0 * PI * nondimensional_extension * factorial_n_minus_2)) * s0)
        }
    }
}

impl Isotensional for FreelyJointedChain {
    /// ```math
    /// \beta\varphi(\eta) = N_b\ln\left(\frac{\eta}{\sinh(\eta)}\right)
    /// ```
    fn nondimensional_gibbs_free_energy(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(self.number_of_links() as Scalar
            * (nondimensional_force / nondimensional_force.sinh()).ln())
    }
    /// ```math
    /// \gamma(\eta) = \mathcal{L}(\eta)
    /// ```
    fn nondimensional_extension(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(langevin(nondimensional_force))
    }
    /// ```math
    /// c(\eta) = \mathcal{L}'(\eta)
    /// ```
    fn nondimensional_compliance(
        &self,
        nondimensional_force: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        Ok(langevin_derivative(nondimensional_force))
    }
}

impl Legendre for FreelyJointedChain {
    /// ```math
    /// \eta(\gamma) = \mathcal{L}^{-1}(\gamma)
    /// ```
    fn nondimensional_force(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        self.nondimensional_extension_check(nondimensional_extension)?;
        Ok(inverse_langevin(nondimensional_extension))
    }
    /// ```math
    /// \mathcal{P}(\gamma) \propto \left\{\frac{\sinh[\eta(\gamma)]}{\eta(\gamma)\exp[\eta(\gamma)\gamma]}\right\}^{N_b}
    /// ```
    fn nondimensional_spherical_distribution(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<Scalar, SingleChainError> {
        let nondimensional_force = Legendre::nondimensional_force(self, nondimensional_extension)?;
        Ok(
            (((nondimensional_force * (1.0 - nondimensional_extension)).exp()
                - (-nondimensional_force * (1.0 + nondimensional_extension)).exp())
                / 2.0
                / nondimensional_force)
                .powi(self.number_of_links() as i32)
                / normalization(self.number_of_links()),
        )
    }
}

fn random_u64() -> u64 {
    let mut value: u64 = 0;
    for _ in 0..8 {
        value = (value << 8) | (crate::get_random() as u64);
    }
    value
}

fn random_uniform() -> f64 {
    // Uniform in [0,1)
    (random_u64() as f64) / (u64::MAX as f64)
}

impl MonteCarlo for FreelyJointedChain {
    fn random_configuration<const N: usize>(&self) -> CurrentCoordinates<N> {
        let mut position = CurrentCoordinate::zero();
        (0..N)
            .map(|_| {
                let cos_theta = 2.0 * random_uniform() - 1.0;
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let phi = TAU * random_uniform();
                position[0] += sin_theta * phi.cos();
                position[1] += sin_theta * phi.sin();
                position[2] += cos_theta;
                position.clone()
            })
            .collect()
    }
}

fn treloar_sums(number_of_links: u8, x: Scalar) -> [Scalar; 3] {
    if number_of_links <= 2 {
        return [Scalar::NAN; 3];
    }
    let n = number_of_links as Scalar;
    let p = (number_of_links - 2) as i32;
    let m = 0.5 * (1.0 - x);
    let k = ((n * m).ceil() as usize)
        .saturating_sub(1)
        .min(number_of_links as usize);
    let k_float = n * m;
    if (k_float - k_float.round()).abs() == 0.0 {
        return [Scalar::NAN; 3];
    }
    let mut binom = 1.0;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for s in 0..=k {
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        let t = m - (s as Scalar) / n;
        let t0 = if p >= 0 {
            t.powi(p)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p)
        };
        let t1 = if p > 0 {
            t.powi(p - 1)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p - 1)
        };
        let t2 = if p > 1 {
            t.powi(p - 2)
        } else if t == 0.0 {
            0.0
        } else {
            t.powi(p - 2)
        };
        s0 += sign * binom * t0;
        s1 += sign * binom * t1;
        s2 += sign * binom * t2;
        let sf = s as Scalar;
        binom *= (n - sf) / (sf + 1.0);
    }
    [s0, s1, s2]
}

fn normalization(number_of_links: u8) -> Scalar {
    match number_of_links {
        0 => Scalar::NAN,
        1 => 1.389_063_303_837_301_3,
        2 => 0.714_480_944_477_587_6,
        3 => 0.446_182_225_454_993_8,
        4 => 0.310_582_574_239_989_03,
        5 => 0.231_583_731_936_937_35,
        6 => 0.181_026_390_997_248_38,
        7 => 0.146_444_713_993_307_1,
        8 => 0.121_590_329_098_661_26,
        9 => 0.103_031_548_251_807_95,
        10 => 0.088_746_746_615_799_2,
        11 => 0.077_477_021_054_147_71,
        12 => 0.068_402_348_281_970_37,
        13 => 0.060_968_329_153_341_53,
        14 => 0.054_788_235_109_506_34,
        15 => 0.049_585_008_268_986_38,
        16 => 0.045_155_543_187_723_454,
        17 => 0.041_347_934_607_350_624,
        18 => 0.038_046_552_454_682_33,
        19 => 0.035_161_995_757_729_23,
        20 => 0.032_624_174_659_916_245,
        21 => 0.030_377_448_781_842_47,
        22 => 0.028_377_147_909_992_65,
        23 => 0.026_587_040_753_179_49,
        24 => 0.024_977_465_826_024_475,
        25 => 0.023_523_932_434_435_773,
        26 => 0.022_206_060_476_346_62,
        27 => 0.021_006_767_816_333_316,
        28 => 0.019_911_640_864_367_884,
        29 => 0.018_908_442_314_802_338,
        30 => 0.017_986_722_687_273_728,
        31 => 0.017_137_511_214_426_318,
        32 => 0.016_353_067_950_360_97,
        33 => 0.015_626_683_526_651_468,
        34 => 0.014_952_516_294_544_284,
        35 => 0.014_325_459_026_026_574,
        36 => 0.013_741_029_152_869_741,
        37 => 0.013_195_277_875_651_702,
        38 => 0.012_684_714_496_711_45,
        39 => 0.012_206_243_109_212_051,
        40 => 0.011_757_109_371_641_886,
        41 => 0.011_334_855_558_612_255,
        42 => 0.010_937_282_437_959_286,
        43 => 0.010_562_416_805_450_496,
        44 => 0.010_208_483_730_069_894,
        45 => 0.009_873_882_738_568_507,
        46 => 0.009_557_167_308_025_053,
        47 => 0.009_257_027_147_393_918,
        48 => 0.008_972_272_839_407_26,
        49 => 0.008_701_822_487_349_997,
        50 => 0.008_444_690_070_700_62,
        51 => 0.008_199_975_262_200_628,
        52 => 0.007_966_854_498_747_187,
        53 => 0.007_744_573_131_302_788,
        54 => 0.007_532_438_506_130_219,
        55 => 0.007_329_813_852_159_101,
        56 => 0.007_136_112_868_025_883,
        57 => 0.006_950_794_917_986_018,
        58 => 0.006_773_360_759_023_209,
        59 => 0.006_603_348_732_523_006,
        60 => 0.006_440_331_363_193_850_5,
        61 => 0.006_283_912_315_803_12,
        62 => 0.006_133_723_666_987_055,
        63 => 0.005_989_423_455_088_637,
        64 => 0.005_850_693_475_837_429,
        65 => 0.005_717_237_295_843_889,
        66 => 0.005_588_778_459_447_456,
        67 => 0.005_465_058_867_524_899,
        68 => 0.005_345_837_309_508_891,
        69 => 0.005_230_888_132_150_507,
        70 => 0.005_120_000_030_536_583,
        71 => 0.005_012_974_948_588_448,
        72 => 0.004_909_627_077_760_272,
        73 => 0.004_809_781_943_954_87,
        74 => 0.004_713_275_573_809_406_5,
        75 => 0.004_619_953_732_495_746,
        76 => 0.004_529_671_226_049_816,
        77 => 0.004_442_291_262_007_673_5,
        78 => 0.004_357_684_862_797_343,
        79 => 0.004_275_730_326_926_852,
        80 => 0.004_196_312_733_530_766,
        81 => 0.004_119_323_486_298_74,
        82 => 0.004_044_659_893_217_922,
        83 => 0.003_972_224_778_923_046,
        84 => 0.003_901_926_126_769_502,
        85 => 0.003_833_676_748_030_511_5,
        86 => 0.003_767_393_975_874_093_7,
        87 => 0.003_702_999_382_002_534_4,
        88 => 0.003_640_418_514_039_760_3,
        89 => 0.003_579_580_651_933_304,
        90 => 0.003_520_418_581_799_841,
        91 => 0.003_462_868_385_788_78,
        92 => 0.003_406_869_246_668_994,
        93 => 0.003_352_363_265_961_134_8,
        94 => 0.003_299_295_294_543_597_2,
        95 => 0.003_247_612_774_755_336_3,
        96 => 0.003_197_265_593_104_502_5,
        97 => 0.003_148_205_942_769_372,
        98 => 0.003_100_388_195_147_996,
        99 => 0.003_053_768_779_776_389_4,
        100 => 0.003_008_306_071_992_423_4,
        101 => 0.002_963_960_287_774_609_6,
        102 => 0.002_920_693_385_232_166,
        103 => 0.002_878_468_972_265_675,
        104 => 0.002_837_252_219_956_577_4,
        105 => 0.002_797_009_781_279_318,
        106 => 0.002_757_709_714_762_243_4,
        107 => 0.002_719_321_412_752_858,
        108 => 0.002_681_815_533_969_964_8,
        109 => 0.002_645_163_940_049_785,
        110 => 0.002_609_339_635_815_636,
        111 => 0.002_574_316_713_021_305,
        112 => 0.002_540_070_297_337_108_2,
        113 => 0.002_506_576_498_364_846,
        114 => 0.002_473_812_362_483_738,
        115 => 0.002_441_755_828_343_935,
        116 => 0.002_410_385_684_837_54,
        117 => 0.002_379_681_531_389_384_6,
        118 => 0.002_349_623_740_421_053,
        119 => 0.002_320_193_421_852_075,
        120 => 0.002_291_372_389_511_768,
        121 => 0.002_263_143_129_344_052_4,
        122 => 0.002_235_488_769_295_701_7,
        123 => 0.002_208_393_050_786_021,
        124 => 0.002_181_840_301_662_884_3,
        125 => 0.002_155_815_410_556_500_4,
        126 => 0.002_130_303_802_548_212_4,
        127 => 0.002_105_291_416_077_128,
        128 => 0.002_080_764_681_012_503,
        129 => 0.002_056_710_497_824_479_8,
        130 => 0.002_033_116_217_790_202_3,
        131 => 0.002_009_969_624_176_372,
        132 => 0.001_987_258_914_343_074,
        133 => 0.001_964_972_682_717_241_4,
        134 => 0.001_943_099_904_587_340_1,
        135 => 0.001_921_629_920_673_915_4,
        136 => 0.001_900_552_422_433_467_6,
        137 => 0.001_879_857_438_055_702_6,
        138 => 0.001_859_535_319_116_718_1,
        139 => 0.001_839_576_727_852_897_4,
        140 => 0.001_819_972_625_022_462_4,
        141 => 0.001_800_714_258_323_577,
        142 => 0.001_781_793_151_339_777_3,
        143 => 0.001_763_201_092_985_212,
        144 => 0.001_744_930_127_423_801_3,
        145 => 0.001_726_972_544_437_928,
        146 => 0.001_709_320_870_223_689,
        147 => 0.001_691_967_858_591_060_8,
        148 => 0.001_674_906_482_548_559_4,
        149 => 0.001_658_129_926_253_140_5,
        150 => 0.001_641_631_577_307_177_5,
        151 => 0.001_625_405_019_385_355_6,
        152 => 0.001_609_444_025_175_287_7,
        153 => 0.001_593_742_549_616_558,
        154 => 0.001_578_294_723_423_718_3,
        155 => 0.001_563_094_846_879_572_4,
        156 => 0.001_548_137_383_885_819,
        157 => 0.001_533_416_956_258_811_4,
        158 => 0.001_518_928_338_258_862_9,
        159 => 0.001_504_666_451_342_125,
        160 => 0.001_490_626_359_124_665_4,
        161 => 0.001_476_803_262_548_898,
        162 => 0.001_463_192_495_243_045,
        163 => 0.001_449_789_519_064_795,
        164 => 0.001_436_589_919_820_758_2,
        165 => 0.001_423_589_403_153_785_4,
        166 => 0.001_410_783_790_590_583,
        167 => 0.001_398_169_015_742_466_7,
        168 => 0.001_385_741_120_652_447_7,
        169 => 0.001_373_496_252_282_184_2,
        170 => 0.001_361_430_659_132_658_5,
        171 => 0.001_349_540_687_992_734_2,
        172 => 0.001_337_822_780_810_052_7,
        173 => 0.001_326_273_471_678_944_4,
        174 => 0.001_314_889_383_940_535_3,
        175 => 0.001_303_667_227_389_772_3,
        176 => 0.001_292_603_795_585_479_2,
        177 => 0.001_281_695_963_258_609_4,
        178 => 0.001_270_940_683_814_773_4,
        179 => 0.001_260_334_986_927_068_6,
        180 => 0.001_249_875_976_215_474_5,
        181 => 0.001_239_560_827_009_231_7,
        182 => 0.001_229_386_784_188_803_6,
        183 => 0.001_219_351_160_104_175_2,
        184 => 0.001_209_451_332_566_384_2,
        185 => 0.001_199_684_742_909_333_8,
        186 => 0.001_190_048_894_119_060_9,
        187 => 0.001_180_541_349_027_766_8,
        188 => 0.001_171_159_728_570_035,
        189 => 0.001_161_901_710_098_784_4,
        190 => 0.001_152_765_025_758_600_2,
        191 => 0.001_143_747_460_914_206_6,
        192 => 0.001_134_846_852_631_931_8,
        193 => 0.001_126_061_088_212_113_2,
        194 => 0.001_117_388_103_770_484_4,
        195 => 0.001_108_825_882_866_665,
        196 => 0.001_100_372_455_177_960_5,
        197 => 0.001_092_025_895_216_747_5,
        198 => 0.001_083_784_321_089_808_8,
        199 => 0.001_075_645_893_298_036_3,
        200 => 0.001_067_608_813_574_995_4,
        201 => 0.001_059_671_323_762_909_2,
        202 => 0.001_051_831_704_724_672_7,
        203 => 0.001_044_088_275_290_578_8,
        204 => 0.001_036_439_391_238_473_6,
        205 => 0.001_028_883_444_306_137,
        206 => 0.001_021_418_861_234_703,
        207 => 0.001_014_044_102_842_014_4,
        208 => 0.001_006_757_663_124_827_5,
        209 => 0.000_999_558_068_388_836_5,
        210 => 0.000_992_443_876_405_530_2,
        211 => 0.000_985_413_675_594_929_3,
        212 => 0.000_978_466_084_233_29,
        213 => 0.000_971_599_749_684_902_6,
        214 => 0.000_964_813_347_657_139_1,
        215 => 0.000_958_105_581_477_943_6,
        216 => 0.000_951_475_181_394_988_1,
        217 => 0.000_944_920_903_895_750_9,
        218 => 0.000_938_441_531_047_793_8,
        219 => 0.000_932_035_869_858_555_4,
        220 => 0.000_925_702_751_653_993_6,
        221 => 0.000_919_441_031_475_440_3,
        222 => 0.000_913_249_587_494_056_1,
        223 => 0.000_907_127_320_442_295_6,
        224 => 0.000_901_073_153_061_812_5,
        225 => 0.000_895_086_029_567_262_1,
        226 => 0.000_889_164_915_125_473_2,
        227 => 0.000_883_308_795_349_485_9,
        228 => 0.000_877_516_675_806_960_8,
        229 => 0.000_871_787_581_542_502_7,
        230 => 0.000_866_120_556_613_433,
        231 => 0.000_860_514_663_638_586_4,
        232 => 0.000_854_968_983_359_706_2,
        233 => 0.000_849_482_614_215_034_8,
        234 => 0.000_844_054_671_924_712_5,
        235 => 0.000_838_684_289_087_606_4,
        236 => 0.000_833_370_614_789_207_2,
        237 => 0.000_828_112_814_220_247_1,
        238 => 0.000_822_910_068_305_699,
        239 => 0.000_817_761_573_343_835,
        240 => 0.000_812_666_540_655_029_1,
        241 => 0.000_807_624_196_240_001_8,
        242 => 0.000_802_633_780_447_217_2,
        243 => 0.000_797_694_547_649_146_8,
        244 => 0.000_792_805_765_927_132_3,
        245 => 0.000_787_966_716_764_583_6,
        246 => 0.000_783_176_694_748_257,
        247 => 0.000_778_435_007_277_372_5,
        248 => 0.000_773_740_974_280_329_3,
        249 => 0.000_769_093_927_938_796_6,
        250 => 0.000_764_493_212_418_953_9,
        251 => 0.000_759_938_183_609_671_9,
        252 => 0.000_755_428_208_867_465_4,
        253 => 0.000_750_962_666_767_783,
        254 => 0.000_746_540_946_863_032_5,
        255 => 0.000_742_162_449_446_367_6,
    }
}
