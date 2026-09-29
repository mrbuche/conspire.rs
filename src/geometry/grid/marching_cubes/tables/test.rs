use super::*;

fn rows<const N: usize>(lut: &Lut<N>) -> usize {
    N / (lut.l1 * lut.l2)
}

fn triangles<const N: usize>(lut: &Lut<N>, index: usize) -> usize {
    (0..)
        .take_while(|&t| 3 * t < lut.l1 && lut.get2(index, 3 * t) != -1)
        .count()
}

#[test]
fn classic_rows_are_terminated_triples_of_edges() {
    for index in 0..256 {
        let row: Vec<i32> = (0..16).map(|j| CASESCLASSIC.get2(index, j)).collect();
        let count = row.iter().take_while(|&&edge| edge != -1).count();
        assert_eq!(count % 3, 0, "index {index}");
        assert!(row[..count].iter().all(|edge| (0..12).contains(edge)));
        assert!(row[count..].iter().all(|&edge| edge == -1), "index {index}");
    }
    assert_eq!(triangles(&CASESCLASSIC, 0), 0);
    assert_eq!(triangles(&CASESCLASSIC, 255), 0);
}

#[test]
fn cases_point_at_rows_of_their_tilings() {
    let primary = [
        rows(&TILING1),
        rows(&TILING2),
        rows(&TILING3_1),
        rows(&TILING4_1),
        rows(&TILING5),
        rows(&TILING6_1_1),
        rows(&TILING7_1),
        rows(&TILING8),
        rows(&TILING9),
        rows(&TILING10_1_1),
        rows(&TILING11),
        rows(&TILING12_1_1),
        rows(&TILING13_1),
        rows(&TILING14),
    ];
    for index in 0..256 {
        let case = CASES.get2(index, 0);
        assert_eq!(case == 0, index == 0 || index == 255, "index {index}");
        if case > 0 {
            assert!(case <= 14);
            assert!(CASES.get2(index, 1) < primary[case as usize - 1] as i32);
        }
    }
}

#[test]
fn unambiguous_tilings_agree_with_the_classic_table() {
    let cases = [(1, 1), (2, 2), (5, 3), (8, 2), (9, 4), (11, 4), (14, 4)];
    for index in 0..256 {
        let case = CASES.get2(index, 0);
        if let Some(&(_, count)) = cases.iter().find(|&&(c, _)| c == case) {
            assert_eq!(triangles(&CASESCLASSIC, index), count, "index {index}");
        }
    }
}

#[test]
fn tilings_hold_only_edges_and_the_centre() {
    let tables: [&[i8]; 38] = [
        &TILING1.values,
        &TILING2.values,
        &TILING3_1.values,
        &TILING3_2.values,
        &TILING4_1.values,
        &TILING4_2.values,
        &TILING5.values,
        &TILING6_1_1.values,
        &TILING6_1_2.values,
        &TILING6_2.values,
        &TILING7_1.values,
        &TILING7_2.values,
        &TILING7_3.values,
        &TILING7_4_1.values,
        &TILING7_4_2.values,
        &TILING8.values,
        &TILING9.values,
        &TILING10_1_1.values,
        &TILING10_1_1_.values,
        &TILING10_1_2.values,
        &TILING10_2.values,
        &TILING10_2_.values,
        &TILING11.values,
        &TILING12_1_1.values,
        &TILING12_1_1_.values,
        &TILING12_1_2.values,
        &TILING12_2.values,
        &TILING12_2_.values,
        &TILING13_1.values,
        &TILING13_1_.values,
        &TILING13_2.values,
        &TILING13_2_.values,
        &TILING13_3.values,
        &TILING13_3_.values,
        &TILING13_4.values,
        &TILING13_5_1.values,
        &TILING13_5_2.values,
        &TILING14.values,
    ];
    for table in tables {
        assert!(table.iter().all(|edge| (-1..=12).contains(edge)));
    }
}

#[test]
fn case_13_face_tests_map_one_to_one_onto_the_possible_subconfigs() {
    assert_eq!(SUBCONFIG13.values.len(), 64);
    let mut possible: Vec<i8> = SUBCONFIG13
        .values
        .iter()
        .copied()
        .filter(|&s| s >= 0)
        .collect();
    possible.sort_unstable();
    assert_eq!(possible, (0..46).collect::<Vec<i8>>());
    assert!(SUBCONFIG13.values.iter().all(|&s| s >= -1));
}
