use super::{cell::Cell, tables::*};

const EPSILON: f64 = f64::EPSILON;

pub(super) fn the_big_switch(cell: &mut Cell, case: i32, config: usize) {
    let mut subconfig = 0;
    match case {
        1 => cell.add_triangles(&TILING1, config, 1),
        2 => cell.add_triangles(&TILING2, config, 2),
        3 => {
            if test_face(cell, TEST3.get1(config)) {
                cell.add_triangles(&TILING3_2, config, 4)
            } else {
                cell.add_triangles(&TILING3_1, config, 2)
            }
        }
        4 => {
            if test_internal(cell, case, config, subconfig, TEST4.get1(config)) {
                cell.add_triangles(&TILING4_1, config, 2)
            } else {
                cell.add_triangles(&TILING4_2, config, 6)
            }
        }
        5 => cell.add_triangles(&TILING5, config, 3),
        6 => {
            if test_face(cell, TEST6.get2(config, 0)) {
                cell.add_triangles(&TILING6_2, config, 5)
            } else if test_internal(cell, case, config, subconfig, TEST6.get2(config, 1)) {
                cell.add_triangles(&TILING6_1_1, config, 3)
            } else {
                cell.add_triangles(&TILING6_1_2, config, 9)
            }
        }
        7 => {
            for bit in 0..3 {
                if test_face(cell, TEST7.get2(config, bit)) {
                    subconfig += 1 << bit;
                }
            }
            match subconfig {
                0 => cell.add_triangles(&TILING7_1, config, 3),
                1 => cell.add_triangles2(&TILING7_2, config, 0, 5),
                2 => cell.add_triangles2(&TILING7_2, config, 1, 5),
                3 => cell.add_triangles2(&TILING7_3, config, 0, 9),
                4 => cell.add_triangles2(&TILING7_2, config, 2, 5),
                5 => cell.add_triangles2(&TILING7_3, config, 1, 9),
                6 => cell.add_triangles2(&TILING7_3, config, 2, 9),
                _ => {
                    if test_internal(cell, case, config, subconfig, TEST7.get2(config, 3)) {
                        cell.add_triangles(&TILING7_4_2, config, 9)
                    } else {
                        cell.add_triangles(&TILING7_4_1, config, 5)
                    }
                }
            }
        }
        8 => cell.add_triangles(&TILING8, config, 2),
        9 => cell.add_triangles(&TILING9, config, 4),
        10 => {
            if test_face(cell, TEST10.get2(config, 0)) {
                if test_face(cell, TEST10.get2(config, 1)) {
                    cell.add_triangles(&TILING10_1_1_, config, 4)
                } else {
                    cell.add_triangles(&TILING10_2, config, 8)
                }
            } else if test_face(cell, TEST10.get2(config, 1)) {
                cell.add_triangles(&TILING10_2_, config, 8)
            } else if test_internal(cell, case, config, subconfig, TEST10.get2(config, 2)) {
                cell.add_triangles(&TILING10_1_1, config, 4)
            } else {
                cell.add_triangles(&TILING10_1_2, config, 8)
            }
        }
        11 => cell.add_triangles(&TILING11, config, 4),
        12 => {
            if test_face(cell, TEST12.get2(config, 0)) {
                if test_face(cell, TEST12.get2(config, 1)) {
                    cell.add_triangles(&TILING12_1_1_, config, 4)
                } else {
                    cell.add_triangles(&TILING12_2, config, 8)
                }
            } else if test_face(cell, TEST12.get2(config, 1)) {
                cell.add_triangles(&TILING12_2_, config, 8)
            } else if test_internal(cell, case, config, subconfig, TEST12.get2(config, 2)) {
                cell.add_triangles(&TILING12_1_1, config, 4)
            } else {
                cell.add_triangles(&TILING12_1_2, config, 8)
            }
        }
        13 => {
            for bit in 0..6 {
                if test_face(cell, TEST13.get2(config, bit)) {
                    subconfig += 1 << bit;
                }
            }
            let subconfig = SUBCONFIG13.get1(subconfig as usize);
            match subconfig {
                0 => cell.add_triangles(&TILING13_1, config, 4),
                1..=6 => cell.add_triangles2(&TILING13_2, config, subconfig as usize - 1, 6),
                7..=18 => cell.add_triangles2(&TILING13_3, config, subconfig as usize - 7, 10),
                19..=22 => cell.add_triangles2(&TILING13_4, config, subconfig as usize - 19, 12),
                23..=26 => {
                    let sub = subconfig as usize - 23;
                    if test_internal(cell, case, config, sub as i32, TEST13.get2(config, 6)) {
                        cell.add_triangles2(&TILING13_5_1, config, sub, 6)
                    } else {
                        cell.add_triangles2(&TILING13_5_2, config, sub, 10)
                    }
                }
                27..=38 => cell.add_triangles2(&TILING13_3_, config, subconfig as usize - 27, 10),
                39..=44 => cell.add_triangles2(&TILING13_2_, config, subconfig as usize - 39, 6),
                45 => cell.add_triangles(&TILING13_1_, config, 4),
                _ => {}
            }
        }
        14 => cell.add_triangles(&TILING14, config, 4),
        _ => {}
    }
}

fn test_face(cell: &Cell, face: i32) -> bool {
    let v = &cell.v;
    let [a, b, c, d] = match face.abs() {
        1 => [v[0], v[4], v[5], v[1]],
        2 => [v[1], v[5], v[6], v[2]],
        3 => [v[2], v[6], v[7], v[3]],
        4 => [v[3], v[7], v[4], v[0]],
        5 => [v[0], v[3], v[2], v[1]],
        6 => [v[4], v[7], v[6], v[5]],
        _ => [0.0; 4],
    };
    let ac_bd = a * c - b * d;
    if ac_bd > -EPSILON && ac_bd < EPSILON {
        face >= 0
    } else {
        f64::from(face) * a * ac_bd >= 0.0
    }
}

fn test_internal(cell: &Cell, case: i32, config: usize, subconfig: i32, s: i32) -> bool {
    let v = &cell.v;
    let (at, bt, ct, dt);
    match case {
        4 | 10 => {
            let a = (v[4] - v[0]) * (v[6] - v[2]) - (v[7] - v[3]) * (v[5] - v[1]);
            let b = v[2] * (v[4] - v[0]) + v[0] * (v[6] - v[2])
                - v[1] * (v[7] - v[3])
                - v[3] * (v[5] - v[1]);
            let t = -b / (2.0 * a + EPSILON);
            if !(0.0..=1.0).contains(&t) {
                return s > 0;
            }
            at = v[0] + (v[4] - v[0]) * t;
            bt = v[3] + (v[7] - v[3]) * t;
            ct = v[2] + (v[6] - v[2]) * t;
            dt = v[1] + (v[5] - v[1]) * t;
        }
        6 | 7 | 12 | 13 => {
            let edge = match case {
                6 => TEST6.get2(config, 2),
                7 => TEST7.get2(config, 4),
                12 => TEST12.get2(config, 3),
                _ => TILING13_5_1.get3(config, subconfig as usize, 0),
            };
            let (t, b, c, d) = match edge {
                0 => (v[0] / (v[0] - v[1] + EPSILON), [3, 2], [7, 6], [4, 5]),
                1 => (v[1] / (v[1] - v[2] + EPSILON), [0, 3], [4, 7], [5, 6]),
                2 => (v[2] / (v[2] - v[3] + EPSILON), [1, 0], [5, 4], [6, 7]),
                3 => (v[3] / (v[3] - v[0] + EPSILON), [2, 1], [6, 5], [7, 4]),
                4 => (v[4] / (v[4] - v[5] + EPSILON), [7, 6], [3, 2], [0, 1]),
                5 => (v[5] / (v[5] - v[6] + EPSILON), [4, 7], [0, 3], [1, 2]),
                6 => (v[6] / (v[6] - v[7] + EPSILON), [5, 4], [1, 0], [2, 3]),
                7 => (v[7] / (v[7] - v[4] + EPSILON), [6, 5], [2, 1], [3, 0]),
                8 => (v[0] / (v[0] - v[4] + EPSILON), [3, 7], [2, 6], [1, 5]),
                9 => (v[1] / (v[1] - v[5] + EPSILON), [0, 4], [3, 7], [2, 6]),
                10 => (v[2] / (v[2] - v[6] + EPSILON), [1, 5], [0, 4], [3, 7]),
                11 => (v[3] / (v[3] - v[7] + EPSILON), [2, 6], [1, 5], [0, 4]),
                _ => (0.0, [0, 0], [0, 0], [0, 0]),
            };
            let along = |[from, to]: [usize; 2]| v[from] + (v[to] - v[from]) * t;
            at = 0.0;
            bt = along(b);
            ct = along(c);
            dt = along(d);
        }
        _ => {
            at = 0.0;
            bt = 0.0;
            ct = 0.0;
            dt = 0.0;
        }
    }
    let mut test = 0;
    if at >= 0.0 {
        test += 1;
    }
    if bt >= 0.0 {
        test += 2;
    }
    if ct >= 0.0 {
        test += 4;
    }
    if dt >= 0.0 {
        test += 8;
    }
    match test {
        0..=4 | 6 | 8 | 9 | 12 => s > 0,
        7 | 11 | 13..=15 => s < 0,
        5 => at * ct - bt * dt < EPSILON && s > 0,
        _ => at * ct - bt * dt >= EPSILON && s > 0,
    }
}
