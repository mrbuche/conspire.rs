#[cfg(test)]
mod test;

use crate::math::Set;

pub struct Sets<R, S, T, U, V>
where
    R: IntoIterator<Item = Set<S, T>>,
    S: IntoIterator<Item = T>,
    U: IntoIterator<Item = V>,
{
    members: R,
    sets: U,
}

impl<S, T> From<Vec<S>> for Sets<Vec<Set<S, T>>, S, T, Vec<usize>, usize>
where
    S: IntoIterator<Item = T>,
{
    fn from(data: Vec<S>) -> Self {
        let sets = (0..data.len()).collect();
        (sets, data).into()
    }
}

impl<S, T, V> From<(Vec<V>, Vec<S>)> for Sets<Vec<Set<S, T>>, S, T, Vec<V>, V>
where
    S: IntoIterator<Item = T>,
{
    fn from((sets, data): (Vec<V>, Vec<S>)) -> Self {
        assert_eq!(data.len(), sets.len());
        let members = data.into_iter().map(|members| Set { members }).collect();
        Self { members, sets }
    }
}

impl<S, T> From<Sets<Vec<Set<S, T>>, S, T, Vec<usize>, usize>> for Vec<S>
where
    S: From<Set<S, T>> + IntoIterator<Item = T>,
{
    fn from(sets: Sets<Vec<Set<S, T>>, S, T, Vec<usize>, usize>) -> Self {
        sets.members
            .into_iter()
            .map(|members| members.into())
            .collect()
    }
}

impl<R, S, T, U, V> Sets<R, S, T, U, V>
where
    R: IntoIterator<Item = Set<S, T>>,
    S: IntoIterator<Item = T>,
    U: IntoIterator<Item = V>,
    for<'a> &'a R: IntoIterator<Item = &'a Set<S, T>>,
    for<'a> &'a S: IntoIterator<Item = &'a T>,
    T: Copy + Ord,
{
    fn unique_members(&self) -> Vec<T> {
        let mut unique_members: Vec<T> = (&self.members)
            .into_iter()
            .flat_map(|members| (&members.members).into_iter().copied())
            .collect();
        unique_members.sort_unstable();
        unique_members.dedup();
        unique_members
    }
}

pub trait InverseSets<R, S, T, U, V>
where
    R: IntoIterator<Item = Set<S, T>>,
    S: IntoIterator<Item = T>,
    U: IntoIterator<Item = V>,
{
    fn inverse(&self) -> Sets<R, S, T, U, V>;
}

impl<R, S, U, V> InverseSets<Vec<Set<Vec<V>, V>>, Vec<V>, V, Vec<usize>, usize>
    for Sets<R, S, usize, U, V>
where
    R: IntoIterator<Item = Set<S, usize>>,
    S: IntoIterator<Item = usize>,
    U: IntoIterator<Item = V>,
    for<'a> &'a R: IntoIterator<Item = &'a Set<S, usize>>,
    for<'a> &'a S: IntoIterator<Item = &'a usize>,
    for<'a> &'a Set<S, usize>: IntoIterator<Item = &'a usize>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    V: Copy,
{
    fn inverse(&self) -> Sets<Vec<Set<Vec<V>, V>>, Vec<V>, V, Vec<usize>, usize> {
        let sets = self.unique_members();
        let mut members = vec![Set { members: vec![] }; sets.len()];
        let max_member = sets.iter().max().unwrap();
        let mut map = vec![0; max_member + 1];
        sets.iter()
            .enumerate()
            .for_each(|(index, &set)| map[set] = index);
        (&self.sets)
            .into_iter()
            .zip(&self.members)
            .for_each(|(&set, set_members)| {
                (&set_members.members)
                    .into_iter()
                    .for_each(|&member| members[map[member]].members.push(set))
            });
        Sets { members, sets }
    }
}
