#[cfg(test)]
mod test;

use super::{
    super::{
        interpolate::InterpolateSolution, Tensor, TensorArray, TensorRank0, TensorVec, Vector,
    },
    Explicit, IntegrationError,
};
use crate::{ABS_TOL, REL_TOL};
use std::ops::{Mul, Sub};

const C_2: TensorRank0 = 0.05;
const C_3: TensorRank0 = 341.0 / 3200.0;
const C_4: TensorRank0 = 1023.0 / 6400.0;
const C_5: TensorRank0 = 39.0 / 100.0;
const C_6: TensorRank0 = 93.0 / 200.0;
const C_7: TensorRank0 = 31.0 / 200.0;
const C_8: TensorRank0 = 943.0 / 1000.0;
const C_9: TensorRank0 = 7067558016280.0 / 7837150160667.0;
const C_10: TensorRank0 = 909.0 / 1000.0;
const C_11: TensorRank0 = 47.0 / 50.0;

const A_2_1: TensorRank0 = 0.05;
const A_3_1: TensorRank0 = -7161.0 / 1024000.0;
const A_3_2: TensorRank0 = 116281.0 / 1024000.0;
const A_4_1: TensorRank0 = 1023.0 / 25600.0;
const A_4_3: TensorRank0 = 3069.0 / 25600.0;
const A_5_1: TensorRank0 = 4202367.0 / 11628100.0;
const A_5_3: TensorRank0 = -3899844.0 / 2907025.0;
const A_5_4: TensorRank0 = 3982992.0 / 2907025.0;
const A_6_1: TensorRank0 = 5611.0 / 114400.0;
const A_6_4: TensorRank0 = 31744.0 / 135025.0;
const A_6_5: TensorRank0 = 923521.0 / 5106400.0;
const A_7_1: TensorRank0 = 21173.0 / 343200.0;
const A_7_4: TensorRank0 = 8602624.0 / 76559175.0;
const A_7_5: TensorRank0 = -26782109.0 / 689364000.0;
const A_7_6: TensorRank0 = 5611.0 / 283500.0;
const A_8_1: TensorRank0 = -1221101821869329.0 / 690812928000000.0;
const A_8_4: TensorRank0 = -125.0 / 2.0;
const A_8_5: TensorRank0 = -1024030607959889.0 / 168929280000000.0;
const A_8_6: TensorRank0 = 1501408353528689.0 / 265697280000000.0;
const A_8_7: TensorRank0 = 6070139212132283.0 / 92502016000000.0;
const A_9_1: TensorRank0 =
    -1472514264486215803881384708877264246346044433307094207829051978044531801133057155.0
        / 1246894801620032001157059621643986024803301558393487900440453636168046069686436608.0;
const A_9_4: TensorRank0 =
    -5172294311085668458375175655246981230039025336933699114138315270772319372469280000.0
        / 124619381004809145897278630571215298365257079410236252921850936749076487132995191.0;
const A_9_5: TensorRank0 =
    -12070679258469254807978936441733187949484571516120469966534514296406891652614970375.0
        / 2722031154761657221710478184531100699497284085048389015085076961673446140398628096.0;
const A_9_6: TensorRank0 =
    780125155843893641323090552530431036567795592568497182701460674803126770111481625.0
        / 183110425412731972197889874507158786859226102980861859505241443073629143100805376.0;
const A_9_7: TensorRank0 =
    664113122959911642134782135839106469928140328160577035357155340392950009492511875.0
        / 15178465598586248136333023107295349175279765150089078301139943253016877823170816.0;
const A_9_8: TensorRank0 = 10332848184452015604056836767286656859124007796970668046446015775000000.0
    / 1312703550036033648073834248740727914537972028638950165249582733679393783.0;
const A_10_1: TensorRank0 = -29055573360337415088538618442231036441314060511.0
    / 22674759891089577691327962602370597632000000000.0;
const A_10_4: TensorRank0 = -20462749524591049105403365239069.0 / 454251913499893469596231268750.0;
const A_10_5: TensorRank0 =
    -180269259803172281163724663224981097.0 / 38100922558256871086579832832000000.0;
const A_10_6: TensorRank0 = 21127670214172802870128286992003940810655221489.0
    / 4679473877997892906145822697976708633673728000.0;
const A_10_7: TensorRank0 = 318607235173649312405151265849660869927653414425413.0
    / 6714716715558965303132938072935465423910912000000.0;
const A_10_8: TensorRank0 =
    212083202434519082281842245535894.0 / 20022426044775672563822865371173879.0;
const A_10_9: TensorRank0 =
    -2698404929400842518721166485087129798562269848229517793703413951226714583.0
        / 469545674913934315077000442080871141884676035902717550325616728175875000000.0;
const A_11_1: TensorRank0 = -2342659845814086836951207140065609179073838476242943917.0
    / 1358480961351056777022231400139158760857532162795520000.0;
const A_11_4: TensorRank0 = -996286030132538159613930889652.0 / 16353068885996164905464325675.0;
const A_11_5: TensorRank0 = -26053085959256534152588089363841.0 / 4377552804565683061011299942400.0;
const A_11_6: TensorRank0 = 20980822345096760292224086794978105312644533925634933539.0
    / 3775889992007550803878727839115494641972212962174156800.0;
const A_11_7: TensorRank0 = 890722993756379186418929622095833835264322635782294899.0
    / 13921242001395112657501941955594013822830119803764736.0;
const A_11_8: TensorRank0 =
    161021426143124178389075121929246710833125.0 / 10997207722131034650667041364346422894371443.0;
const A_11_9: TensorRank0 = 300760669768102517834232497565452434946672266195876496371874262392684852243925359864884962513.0 /
    4655443337501346455585065336604505603760824779615521285751892810315680492364106674524398280000.0;
const A_11_10: TensorRank0 = -31155237437111730665923206875.0 / 392862141594230515010338956291.0;
const A_12_1: TensorRank0 = -2866556991825663971778295329101033887534912787724034363.0
    / 868226711619262703011213925016143612030669233795338240.0;
const A_12_4: TensorRank0 =
    -16957088714171468676387054358954754000.0 / 143690415119654683326368228101570221.0;
const A_12_5: TensorRank0 =
    -4583493974484572912949314673356033540575.0 / 451957703655250747157313034270335135744.0;
const A_12_6: TensorRank0 = 2346305388553404258656258473446184419154740172519949575.0
    / 256726716407895402892744978301151486254183185289662464.0;
const A_12_7: TensorRank0 = 1657121559319846802171283690913610698586256573484808662625.0
    / 13431480411255146477259155104956093505361644432088109056.0;
const A_12_8: TensorRank0 = 345685379554677052215495825476969226377187500.0
    / 74771167436930077221667203179551347546362089.0;
const A_12_9: TensorRank0 = -3205890962717072542791434312152727534008102774023210240571361570757249056167015230160352087048674542196011.0 /
    947569549683965814783015124451273604984657747127257615372449205973192657306017239103491074738324033259120.0;
const A_12_10: TensorRank0 =
    40279545832706233433100438588458933210937500.0 / 8896460842799482846916972126377338947215101.0;
const A_12_11: TensorRank0 =
    -6122933601070769591613093993993358877250.0 / 1050517001510235513198246721302027675953.0;
const A_13_1: TensorRank0 = -618675905535482500672800859344538410358660153899637.0
    / 203544282118214047100119475340667684874292102389760.0;
const A_13_4: TensorRank0 =
    -4411194916804718600478400319122931000.0 / 40373053902469967450761491269633019.0;
const A_13_5: TensorRank0 =
    -16734711409449292534539422531728520225.0 / 1801243715290088669307203927210237952.0;
const A_13_6: TensorRank0 = 135137519757054679098042184152749677761254751865630525.0
    / 16029587794486289597771326361911895112703716593983488.0;
const A_13_7: TensorRank0 = 38937568367409876012548551903492196137929710431584875.0
    / 340956454090191606099548798001469306974758443147264.0;
const A_13_8: TensorRank0 =
    -6748865855011993037732355335815350667265625.0 / 7002880395717424621213565406715087764770357.0;
const A_13_9: TensorRank0 = -1756005520307450928195422767042525091954178296002788308926563193523662404739779789732685671.0 /
                             348767814578469983605688098046186480904607278021030540735333862087061574934154942830062320.0;
const A_13_10: TensorRank0 =
    53381024589235611084013897674181629296875.0 / 8959357584795694524874969598508592944141.0;

const B_1: TensorRank0 = 44901867737754616851973.0 / 1014046409980231013380680.0;
const B_6: TensorRank0 = 791638675191615279648100000.0 / 2235604725089973126411512319.0;
const B_7: TensorRank0 = 3847749490868980348119500000.0 / 15517045062138271618141237517.0;
const B_8: TensorRank0 = -13734512432397741476562500000.0 / 875132892924995907746928783.0;
const B_9: TensorRank0 = 12274765470313196878428812037740635050319234276006986398294443554969616342274215316330684448207141.0 / 489345147493715517650385834143510934888829280686609654482896526796523353052166757299452852166040.0;
const B_10: TensorRank0 = -9798363684577739445312500000.0 / 308722986341456031822630699.0;
const B_11: TensorRank0 = 282035543183190840068750.0 / 12295407629873040425991.0;
const B_12: TensorRank0 = -306814272936976936753.0 / 1299331183183744997286.0;

const D_1: TensorRank0 = -225628434546552672055.0 / 6895515587865570890988624.0;
const D_6: TensorRank0 = -1128142172732763360275000.0 / 2235604725089973126411512319.0;
const D_7: TensorRank0 = 5640710863663816801375000.0 / 46551135186414814854423712551.0;
const D_8: TensorRank0 = -17627221448949427504296875000.0 / 875132892924995907746928783.0;
const D_9: TensorRank0 = 17426957952517932078050241885889670195876481434157580946550703126433816616672116622859678756257765.0 / 3327547002957265520022623672175874357244039108668945650483696382216358800754733949636279394729072.0;
const D_10: TensorRank0 = -17627221448949427504296875000.0 / 2161060904390192222758414893.0;
const D_11: TensorRank0 = 282035543183190840068750.0 / 12295407629873040425991.0;
const D_12: TensorRank0 = -306814272936976936753.0 / 1299331183183744997286.0;
const D_13: TensorRank0 = 28735456870978964189.0 / 79783493704265043693.0;

const E_1: TensorRank0 = B_1 - D_1;
const E_6: TensorRank0 = B_6 - D_6;
const E_7: TensorRank0 = B_7 - D_7;
const E_8: TensorRank0 = B_8 - D_8;
const E_9: TensorRank0 = B_9 - D_9;
const E_10: TensorRank0 = B_10 - D_10;
const E_11: TensorRank0 = B_11 - D_11;
const E_12: TensorRank0 = B_12 - D_12;
const E_13: TensorRank0 = -D_13;

// guess coefficients from bhat
// error coefficients are from (bhat - btilde)

// https://www.sfu.ca/~jverner/RKV87.IIa.Efficient.000000282866.081208.CoeffsOnlyFLOAT
// https://www.sfu.ca/~jverner/RKV87.IIa.Robust.00000754677.081208.CoeffsOnlyRATandFLOAT
// https://github.com/SciML/OrdinaryDiffEq.jl/blob/ad7891e95d8907b82adb31b5fbaa0d2d7d38a791/lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl#L2030

// https://github.com/SciML/OrdinaryDiffEq.jl/blob/ad7891e95d8907b82adb31b5fbaa0d2d7d38a791/lib/OrdinaryDiffEqExplicitRK/src/algorithms.jl
// https://numerary.readthedocs.io/en/latest/dormand-prince-method.html
// https://mrbuche.github.io/conspire.rs/latest/math/integrate/struct.Ode45.html

/// Explicit, thirteen-stage, eighth-order, variable-step, Runge-Kutta method.[^cite]
///
/// [^cite]: J.H. Verner, [Numer. Algorithms **53**, 383 (2010)](https://doi.org/10.1007/s11075-009-9290-3).
#[derive(Debug)]
pub struct Ode78 {
    /// Absolute error tolerance.
    pub abs_tol: TensorRank0,
    /// Relative error tolerance.
    pub rel_tol: TensorRank0,
    /// Multiplier for adaptive time steps.
    pub dt_beta: TensorRank0,
    /// Exponent for adaptive time steps.
    pub dt_expn: TensorRank0,
    /// Initial relative time step.
    pub dt_init: TensorRank0,
}

impl Default for Ode78 {
    fn default() -> Self {
        Self {
            abs_tol: ABS_TOL,
            rel_tol: REL_TOL,
            dt_beta: 0.9,
            dt_expn: 8.0,
            dt_init: 0.1,
        }
    }
}

impl<Y, U> Explicit<Y, U> for Ode78
where
    Self: InterpolateSolution<Y, U>,
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn integrate(
        &self,
        function: impl Fn(&TensorRank0, &Y) -> Y,
        initial_condition: Y,
        time: &[TensorRank0],
    ) -> Result<(Vector, U), IntegrationError> {
        if time.len() < 2 {
            return Err(IntegrationError::LengthTimeLessThanTwo);
        } else if time[0] >= time[time.len() - 1] {
            return Err(IntegrationError::InitialTimeNotLessThanFinalTime);
        }
        let mut t = time[0];
        let mut dt = self.dt_init * time[time.len() - 1];
        let mut e;
        let mut k_1;
        let mut k_2;
        let mut k_3;
        let mut k_4;
        let mut k_5;
        let mut k_6;
        let mut k_7;
        let mut k_8;
        let mut k_9;
        let mut k_10;
        let mut k_11;
        let mut k_12;
        let mut k_13;
        let mut t_sol = Vector::zero(0);
        t_sol.push(time[0]);
        let mut y = initial_condition.copy();
        let mut y_sol = U::zero(0);
        y_sol.push(initial_condition.copy());
        let mut y_trial;
        while t < time[time.len() - 1] {
            k_1 = function(&t, &y);
            k_2 = function(&(t + C_2 * dt), &(&k_1 * (A_2_1 * dt) + &y));
            k_3 = function(
                &(t + C_3 * dt),
                &(&k_1 * (A_3_1 * dt) + &k_2 * (A_3_2 * dt) + &y),
            );
            k_4 = function(
                &(t + C_4 * dt),
                &(&k_1 * (A_4_1 * dt) + &k_3 * (A_4_3 * dt) + &y),
            );
            k_5 = function(
                &(t + C_5 * dt),
                &(&k_1 * (A_5_1 * dt) + &k_3 * (A_5_3 * dt) + &k_4 * (A_5_4 * dt) + &y),
            );
            k_6 = function(
                &(t + C_6 * dt),
                &(&k_1 * (A_6_1 * dt) + &k_4 * (A_6_4 * dt) + &k_5 * (A_6_5 * dt) + &y),
            );
            k_7 = function(
                &(t + C_7 * dt),
                &(&k_1 * (A_7_1 * dt)
                    + &k_4 * (A_7_4 * dt)
                    + &k_5 * (A_7_5 * dt)
                    + &k_6 * (A_7_6 * dt)
                    + &y),
            );
            k_8 = function(
                &(t + C_8 * dt),
                &(&k_1 * (A_8_1 * dt)
                    + &k_4 * (A_8_4 * dt)
                    + &k_5 * (A_8_5 * dt)
                    + &k_6 * (A_8_6 * dt)
                    + &k_7 * (A_8_7 * dt)
                    + &y),
            );
            k_9 = function(
                &(t + C_9 * dt),
                &(&k_1 * (A_9_1 * dt)
                    + &k_4 * (A_9_4 * dt)
                    + &k_5 * (A_9_5 * dt)
                    + &k_6 * (A_9_6 * dt)
                    + &k_7 * (A_9_7 * dt)
                    + &k_8 * (A_9_8 * dt)
                    + &y),
            );
            k_10 = function(
                &(t + C_10 * dt),
                &(&k_1 * (A_10_1 * dt)
                    + &k_4 * (A_10_4 * dt)
                    + &k_5 * (A_10_5 * dt)
                    + &k_6 * (A_10_6 * dt)
                    + &k_7 * (A_10_7 * dt)
                    + &k_8 * (A_10_8 * dt)
                    + &k_9 * (A_10_9 * dt)
                    + &y),
            );
            k_11 = function(
                &(t + C_11 * dt),
                &(&k_1 * (A_11_1 * dt)
                    + &k_4 * (A_11_4 * dt)
                    + &k_5 * (A_11_5 * dt)
                    + &k_6 * (A_11_6 * dt)
                    + &k_7 * (A_11_7 * dt)
                    + &k_8 * (A_11_8 * dt)
                    + &k_9 * (A_11_9 * dt)
                    + &k_10 * (A_11_10 * dt)
                    + &y),
            );
            k_12 = function(
                &(t + dt),
                &(&k_1 * (A_12_1 * dt)
                    + &k_4 * (A_12_4 * dt)
                    + &k_5 * (A_12_5 * dt)
                    + &k_6 * (A_12_6 * dt)
                    + &k_7 * (A_12_7 * dt)
                    + &k_8 * (A_12_8 * dt)
                    + &k_9 * (A_12_9 * dt)
                    + &k_10 * (A_12_10 * dt)
                    + &k_11 * (A_12_11 * dt)
                    + &y),
            );
            k_13 = function(
                &(t + dt),
                &(&k_1 * (A_13_1 * dt)
                    + &k_4 * (A_13_4 * dt)
                    + &k_5 * (A_13_5 * dt)
                    + &k_6 * (A_13_6 * dt)
                    + &k_7 * (A_13_7 * dt)
                    + &k_8 * (A_13_8 * dt)
                    + &k_9 * (A_13_9 * dt)
                    + &k_10 * (A_13_10 * dt)
                    + &y),
            );
            y_trial = (&k_1 * B_1
                + &k_6 * B_6
                + &k_7 * B_7
                + &k_8 * B_8
                + &k_9 * B_9
                + &k_10 * B_10
                + &k_11 * B_11
                + &k_12 * B_12)
                * dt
                + &y;
            e = ((&k_1 * E_1
                + &k_6 * E_6
                + &k_7 * E_7
                + &k_8 * E_8
                + &k_9 * E_9
                + &k_10 * E_10
                + &k_11 * E_11
                + &k_12 * E_12
                + &k_13 * E_13)
                * dt)
                .norm();
            if e < self.abs_tol || e / y_trial.norm() < self.rel_tol {
                t += dt;
                y = y_trial;
                t_sol.push(t.copy());
                y_sol.push(y.copy());
            }
            dt *= self.dt_beta * (self.abs_tol / e).powf(1.0 / self.dt_expn);
        }
        if time.len() > 2 {
            let t_int = Vector::new(time);
            let y_int = self.interpolate(&t_int, &t_sol, &y_sol, function);
            Ok((t_int, y_int))
        } else {
            Ok((t_sol, y_sol))
        }
    }
}

impl<Y, U> InterpolateSolution<Y, U> for Ode78
where
    Y: Tensor + TensorArray,
    for<'a> &'a Y: Mul<TensorRank0, Output = Y> + Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
{
    fn interpolate(
        &self,
        ti: &Vector,
        tp: &Vector,
        yp: &U,
        f: impl Fn(&TensorRank0, &Y) -> Y,
    ) -> U {
        todo!()
    }
}
