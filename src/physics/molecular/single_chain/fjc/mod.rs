#[cfg(test)]
mod test;

use crate::{
    math::{
        Scalar,
        special::{inverse_langevin, langevin, langevin_derivative},
    },
    physics::molecular::single_chain::{
        Ensemble, Inextensible, Isometric, Isotensional, Legendre, SingleChain, SingleChainError,
        Thermodynamics,
    },
};
use std::f64::consts::PI;

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
        1 => 1.3890633038373013,
        2 => 0.7144809444775876,
        3 => 0.4461822254549938,
        4 => 0.31058257423998903,
        5 => 0.23158373193693735,
        6 => 0.18102639099724838,
        7 => 0.1464447139933071,
        8 => 0.12159032909866126,
        9 => 0.10303154825180795,
        10 => 0.0887467466157992,
        11 => 0.07747702105414771,
        12 => 0.06840234828197037,
        13 => 0.06096832915334153,
        14 => 0.05478823510950634,
        15 => 0.04958500826898638,
        16 => 0.045155543187723454,
        17 => 0.041347934607350624,
        18 => 0.03804655245468233,
        19 => 0.03516199575772923,
        20 => 0.032624174659916245,
        21 => 0.03037744878184247,
        22 => 0.02837714790999265,
        23 => 0.02658704075317949,
        24 => 0.024977465826024475,
        25 => 0.023523932434435773,
        26 => 0.02220606047634662,
        27 => 0.021006767816333316,
        28 => 0.019911640864367884,
        29 => 0.018908442314802338,
        30 => 0.017986722687273728,
        31 => 0.017137511214426318,
        32 => 0.01635306795036097,
        33 => 0.015626683526651468,
        34 => 0.014952516294544284,
        35 => 0.014325459026026574,
        36 => 0.013741029152869741,
        37 => 0.013195277875651702,
        38 => 0.01268471449671145,
        39 => 0.012206243109212051,
        40 => 0.011757109371641886,
        41 => 0.011334855558612255,
        42 => 0.010937282437959286,
        43 => 0.010562416805450496,
        44 => 0.010208483730069894,
        45 => 0.009873882738568507,
        46 => 0.009557167308025053,
        47 => 0.009257027147393918,
        48 => 0.00897227283940726,
        49 => 0.008701822487349997,
        50 => 0.00844469007070062,
        51 => 0.008199975262200628,
        52 => 0.007966854498747187,
        53 => 0.007744573131302788,
        54 => 0.007532438506130219,
        55 => 0.007329813852159101,
        56 => 0.007136112868025883,
        57 => 0.006950794917986018,
        58 => 0.006773360759023209,
        59 => 0.006603348732523006,
        60 => 0.0064403313631938505,
        61 => 0.00628391231580312,
        62 => 0.006133723666987055,
        63 => 0.005989423455088637,
        64 => 0.005850693475837429,
        65 => 0.005717237295843889,
        66 => 0.005588778459447456,
        67 => 0.005465058867524899,
        68 => 0.005345837309508891,
        69 => 0.005230888132150507,
        70 => 0.005120000030536583,
        71 => 0.005012974948588448,
        72 => 0.004909627077760272,
        73 => 0.00480978194395487,
        74 => 0.0047132755738094065,
        75 => 0.004619953732495746,
        76 => 0.004529671226049816,
        77 => 0.0044422912620076735,
        78 => 0.004357684862797343,
        79 => 0.004275730326926852,
        80 => 0.004196312733530766,
        81 => 0.00411932348629874,
        82 => 0.004044659893217922,
        83 => 0.003972224778923046,
        84 => 0.003901926126769502,
        85 => 0.0038336767480305115,
        86 => 0.0037673939758740937,
        87 => 0.0037029993820025344,
        88 => 0.0036404185140397603,
        89 => 0.003579580651933304,
        90 => 0.003520418581799841,
        91 => 0.00346286838578878,
        92 => 0.003406869246668994,
        93 => 0.0033523632659611348,
        94 => 0.0032992952945435972,
        95 => 0.0032476127747553363,
        96 => 0.0031972655931045025,
        97 => 0.003148205942769372,
        98 => 0.003100388195147996,
        99 => 0.0030537687797763894,
        100 => 0.0030083060719924234,
        101 => 0.0029639602877746096,
        102 => 0.002920693385232166,
        103 => 0.002878468972265675,
        104 => 0.0028372522199565774,
        105 => 0.002797009781279318,
        106 => 0.0027577097147622434,
        107 => 0.002719321412752858,
        108 => 0.0026818155339699648,
        109 => 0.002645163940049785,
        110 => 0.002609339635815636,
        111 => 0.002574316713021305,
        112 => 0.0025400702973371082,
        113 => 0.002506576498364846,
        114 => 0.002473812362483738,
        115 => 0.002441755828343935,
        116 => 0.00241038568483754,
        117 => 0.0023796815313893846,
        118 => 0.002349623740421053,
        119 => 0.002320193421852075,
        120 => 0.002291372389511768,
        121 => 0.0022631431293440524,
        122 => 0.0022354887692957017,
        123 => 0.002208393050786021,
        124 => 0.0021818403016628843,
        125 => 0.0021558154105565004,
        126 => 0.0021303038025482124,
        127 => 0.002105291416077128,
        128 => 0.002080764681012503,
        129 => 0.0020567104978244798,
        130 => 0.0020331162177902023,
        131 => 0.002009969624176372,
        132 => 0.001987258914343074,
        133 => 0.0019649726827172414,
        134 => 0.0019430999045873401,
        135 => 0.0019216299206739154,
        136 => 0.0019005524224334676,
        137 => 0.0018798574380557026,
        138 => 0.0018595353191167181,
        139 => 0.0018395767278528974,
        140 => 0.0018199726250224624,
        141 => 0.001800714258323577,
        142 => 0.0017817931513397773,
        143 => 0.001763201092985212,
        144 => 0.0017449301274238013,
        145 => 0.001726972544437928,
        146 => 0.001709320870223689,
        147 => 0.0016919678585910608,
        148 => 0.0016749064825485594,
        149 => 0.0016581299262531405,
        150 => 0.0016416315773071775,
        151 => 0.0016254050193853556,
        152 => 0.0016094440251752877,
        153 => 0.001593742549616558,
        154 => 0.0015782947234237183,
        155 => 0.0015630948468795724,
        156 => 0.001548137383885819,
        157 => 0.0015334169562588114,
        158 => 0.0015189283382588629,
        159 => 0.001504666451342125,
        160 => 0.0014906263591246654,
        161 => 0.001476803262548898,
        162 => 0.001463192495243045,
        163 => 0.001449789519064795,
        164 => 0.0014365899198207582,
        165 => 0.0014235894031537854,
        166 => 0.001410783790590583,
        167 => 0.0013981690157424667,
        168 => 0.0013857411206524477,
        169 => 0.0013734962522821842,
        170 => 0.0013614306591326585,
        171 => 0.0013495406879927342,
        172 => 0.0013378227808100527,
        173 => 0.0013262734716789444,
        174 => 0.0013148893839405353,
        175 => 0.0013036672273897723,
        176 => 0.0012926037955854792,
        177 => 0.0012816959632586094,
        178 => 0.0012709406838147734,
        179 => 0.0012603349869270686,
        180 => 0.0012498759762154745,
        181 => 0.0012395608270092317,
        182 => 0.0012293867841888036,
        183 => 0.0012193511601041752,
        184 => 0.0012094513325663842,
        185 => 0.0011996847429093338,
        186 => 0.0011900488941190609,
        187 => 0.0011805413490277668,
        188 => 0.001171159728570035,
        189 => 0.0011619017100987844,
        190 => 0.0011527650257586002,
        191 => 0.0011437474609142066,
        192 => 0.0011348468526319318,
        193 => 0.0011260610882121132,
        194 => 0.0011173881037704844,
        195 => 0.001108825882866665,
        196 => 0.0011003724551779605,
        197 => 0.0010920258952167475,
        198 => 0.0010837843210898088,
        199 => 0.0010756458932980363,
        200 => 0.0010676088135749954,
        201 => 0.0010596713237629092,
        202 => 0.0010518317047246727,
        203 => 0.0010440882752905788,
        204 => 0.0010364393912384736,
        205 => 0.001028883444306137,
        206 => 0.001021418861234703,
        207 => 0.0010140441028420144,
        208 => 0.0010067576631248275,
        209 => 0.0009995580683888365,
        210 => 0.0009924438764055302,
        211 => 0.0009854136755949293,
        212 => 0.00097846608423329,
        213 => 0.0009715997496849026,
        214 => 0.0009648133476571391,
        215 => 0.0009581055814779436,
        216 => 0.0009514751813949881,
        217 => 0.0009449209038957509,
        218 => 0.0009384415310477938,
        219 => 0.0009320358698585554,
        220 => 0.0009257027516539936,
        221 => 0.0009194410314754403,
        222 => 0.0009132495874940561,
        223 => 0.0009071273204422956,
        224 => 0.0009010731530618125,
        225 => 0.0008950860295672621,
        226 => 0.0008891649151254732,
        227 => 0.0008833087953494859,
        228 => 0.0008775166758069608,
        229 => 0.0008717875815425027,
        230 => 0.000866120556613433,
        231 => 0.0008605146636385864,
        232 => 0.0008549689833597062,
        233 => 0.0008494826142150348,
        234 => 0.0008440546719247125,
        235 => 0.0008386842890876064,
        236 => 0.0008333706147892072,
        237 => 0.0008281128142202471,
        238 => 0.000822910068305699,
        239 => 0.000817761573343835,
        240 => 0.0008126665406550291,
        241 => 0.0008076241962400018,
        242 => 0.0008026337804472172,
        243 => 0.0007976945476491468,
        244 => 0.0007928057659271323,
        245 => 0.0007879667167645836,
        246 => 0.000783176694748257,
        247 => 0.0007784350072773725,
        248 => 0.0007737409742803293,
        249 => 0.0007690939279387966,
        250 => 0.0007644932124189539,
        251 => 0.0007599381836096719,
        252 => 0.0007554282088674654,
        253 => 0.000750962666767783,
        254 => 0.0007465409468630325,
        255 => 0.0007421624494463676,
    }
}
