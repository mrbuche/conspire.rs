use crate::geometry::ntree::{
    Orthotree,
    leaf::{Leaf, morton::Morton, split::Split},
};
use std::{
    array::from_fn,
    collections::{BTreeMap, VecDeque},
    ops::{AddAssign, Rem, SubAssign},
};

pub enum Balancing {
    Faces,
    All,
}

impl<const D: usize, const N: usize, T, U> Orthotree<D, N, T, U>
where
    T: AddAssign
        + Copy
        + Default
        + Into<u64>
        + PartialEq
        + PartialOrd
        + Rem<Output = T>
        + Split
        + SubAssign,
    U: Copy,
{
    pub fn balance(&mut self, balancing: Balancing) {
        let dirs = generate_directions::<D>(balancing);
        let mut map: BTreeMap<u64, Leaf<D, T, U>> = std::mem::take(&mut self.leaves)
            .into_iter()
            .map(|l| (l.corner.morton(), l))
            .collect();
        let mut queue: VecDeque<u64> = map.keys().copied().collect();
        while let Some(key) = queue.pop_front() {
            let leaf = match map.get(&key).copied() {
                Some(l) => l,
                None => continue,
            };
            let corner = leaf.corner;
            let length = leaf.length;
            let mut double_length = length;
            double_length += length;
            'dir: for dir in &dirs {
                let mut neighbor_corner = corner;
                for (axis, &d) in dir.iter().enumerate() {
                    match d {
                        1 => neighbor_corner[axis] += length,
                        -1 => {
                            if corner[axis] == T::default() {
                                continue 'dir;
                            }
                            neighbor_corner[axis] -= length;
                        }
                        _ => {}
                    }
                }
                // Inner loop: keep subdividing the neighbor in this direction until
                // the 2:1 condition is satisfied. Without this, a fine leaf adjacent
                // to a coarse neighbor at 8:1 or 16:1 only triggers one subdivision
                // and the remaining gap goes undetected.
                loop {
                    let target = neighbor_corner.morton();
                    match map.range(..=target).next_back() {
                        Some((&_nk, &nleaf)) => {
                            let contained = (0..D).all(|ax| {
                                let mut end = nleaf.corner[ax];
                                end += nleaf.length;
                                nleaf.corner[ax] <= neighbor_corner[ax]
                                    && neighbor_corner[ax] < end
                            });
                            if contained && nleaf.length > double_length {
                                let new_keys = subdivide_pairing::<D, N, T, U>(&mut map, nleaf);
                                queue.extend(new_keys);
                                // loop: re-check this direction after subdivision
                            } else {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }
        }
        self.leaves = map.into_values().collect();
    }
}

fn subdivide_pairing<const D: usize, const N: usize, T, U>(
    map: &mut BTreeMap<u64, Leaf<D, T, U>>,
    leaf: Leaf<D, T, U>,
) -> Vec<u64>
where
    T: AddAssign + Copy + Default + Into<u64> + PartialEq + Rem<Output = T> + Split + SubAssign,
    U: Copy,
{
    let mut new_keys = Vec::new();
    let length = leaf.length;
    let corner = leaf.corner;
    let mut parent_length = length;
    parent_length += length;
    let parent_corner: [T; D] = from_fn(|ax| {
        if corner[ax] % parent_length == T::default() {
            corner[ax]
        } else {
            let mut pc = corner[ax];
            pc -= length;
            pc
        }
    });
    let orthant_length = length.split();
    for i in 0..N {
        let sc: [T; D] = from_fn(|ax| {
            if (i >> ax) & 1 == 1 {
                let mut c = parent_corner[ax];
                c += length;
                c
            } else {
                parent_corner[ax]
            }
        });
        let sc_key = sc.morton();
        if let Some(sib) = map.get(&sc_key).copied()
            && sib.length == length
        {
            let data = sib.data;
            map.remove(&sc_key);
            for j in 0..N {
                let oc: [T; D] = from_fn(|ax| {
                    if (j >> ax) & 1 == 1 {
                        let mut c = sc[ax];
                        c += orthant_length;
                        c
                    } else {
                        sc[ax]
                    }
                });
                let ok = oc.morton();
                map.insert(ok, Leaf { corner: oc, length: orthant_length, data });
                new_keys.push(ok);
            }
        }
    }
    new_keys
}

fn generate_directions<const D: usize>(balancing: Balancing) -> Vec<[i8; D]> {
    let mut dirs = Vec::new();
    for i in 0..3_usize.pow(D as u32) {
        let mut dir = [0i8; D];
        let mut val = i;
        for d in dir.iter_mut() {
            *d = (val % 3) as i8 - 1;
            val /= 3;
        }
        let nonzero = dir.iter().filter(|&&d| d != 0).count();
        match balancing {
            Balancing::Faces if nonzero == 1 => dirs.push(dir),
            Balancing::All if nonzero >= 1 => dirs.push(dir),
            _ => {}
        }
    }
    dirs
}
