//! Floating-point expansions: sums of non-overlapping `f64`s that hold a real number exactly.[^1]
//!
//! [^1]: J. R. Shewchuk, [Discrete Comput. Geom. **18**, 305 (1997)](https://doi.org/10.1007/PL00009321).

#[cfg(test)]
mod test;

use core::cmp::Ordering;

/// A real number as non-overlapping components, no zeros, in order of increasing magnitude.
///
/// Arithmetic here is exact unless a component overflows or underflows.
#[derive(Clone, Debug)]
pub(super) struct Expansion(Vec<f64>);

/// `a + b` as the rounded sum and the exact rounding error.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let virtual_b = sum - a;
    let virtual_a = sum - virtual_b;
    (sum, (a - virtual_a) + (b - virtual_b))
}

/// `a + b` as the rounded sum and the exact rounding error, for `|a| >= |b|`.
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}

/// `a * b` as the rounded product and the exact rounding error.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, a.mul_add(b, -product))
}

impl Expansion {
    pub fn new(value: f64) -> Self {
        Self(if value == 0.0 {
            Vec::new()
        } else {
            vec![value]
        })
    }
    /// `a - b`, exactly.
    pub fn difference(a: f64, b: f64) -> Self {
        let (sum, error) = two_sum(a, -b);
        Self::from_pair(error, sum)
    }
    /// `a * b`, exactly.
    #[cfg(test)]
    pub fn product(a: f64, b: f64) -> Self {
        let (product, error) = two_product(a, b);
        Self::from_pair(error, product)
    }
    fn from_pair(small: f64, large: f64) -> Self {
        Self([small, large].into_iter().filter(|&x| x != 0.0).collect())
    }
    pub fn sum(self, other: Self) -> Self {
        other
            .0
            .into_iter()
            .fold(self, |expansion, component| expansion.grown(component))
    }
    pub fn minus(self, other: Self) -> Self {
        self.sum(other.negated())
    }
    pub fn times(&self, other: &Self) -> Self {
        other
            .0
            .iter()
            .fold(Self(Vec::new()), |expansion, &component| {
                expansion.sum(self.scaled(component))
            })
    }
    pub fn negated(self) -> Self {
        Self(self.0.into_iter().map(|component| -component).collect())
    }
    /// The sign of the number, which is that of the largest component.
    pub fn sign(&self) -> Ordering {
        self.0
            .last()
            .map_or(Ordering::Equal, |component| component.total_cmp(&0.0))
    }
    /// The number as a float, which is within one rounding of it.
    pub fn estimate(&self) -> f64 {
        self.0.iter().sum()
    }
    /// The number again, in as few components as it takes, which are then also nonadjacent.[^1]
    ///
    /// Products and sums grow the expansion by more components than the number needs, and every
    /// operation after that pays for them.
    ///
    /// [^1]: J. R. Shewchuk, [Discrete Comput. Geom. **18**, 305 (1997)](https://doi.org/10.1007/PL00009321).
    pub fn compressed(self) -> Self {
        let components = self.0;
        let length = components.len();
        if length < 2 {
            return Self(components);
        }
        let mut gathered = vec![0.0; length];
        let mut bottom = length - 1;
        let mut carry = components[length - 1];
        for &component in components[..length - 1].iter().rev() {
            let (sum, error) = fast_two_sum(carry, component);
            if error != 0.0 {
                gathered[bottom] = sum;
                bottom -= 1;
                carry = error;
            } else {
                carry = sum;
            }
        }
        gathered[bottom] = carry;
        let mut top = 0;
        carry = gathered[bottom];
        for index in bottom + 1..length {
            let (sum, error) = fast_two_sum(gathered[index], carry);
            if error != 0.0 {
                gathered[top] = error;
                top += 1;
            }
            carry = sum;
        }
        gathered[top] = carry;
        gathered.truncate(top + 1);
        gathered.retain(|&component| component != 0.0);
        Self(gathered)
    }
    /// The number of components.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// The sum with one more number.
    fn grown(self, value: f64) -> Self {
        let mut sum = value;
        let mut components = Vec::with_capacity(self.0.len() + 1);
        for component in self.0 {
            let (next, error) = two_sum(sum, component);
            if error != 0.0 {
                components.push(error);
            }
            sum = next;
        }
        if sum != 0.0 {
            components.push(sum);
        }
        Self(components)
    }
    /// The product with one number.
    fn scaled(&self, value: f64) -> Self {
        let Some(&first) = self.0.first() else {
            return Self(Vec::new());
        };
        let (mut sum, error) = two_product(first, value);
        let mut components = Vec::with_capacity(2 * self.0.len());
        if error != 0.0 {
            components.push(error);
        }
        for &component in &self.0[1..] {
            let (product, product_error) = two_product(component, value);
            let (next, error) = two_sum(sum, product_error);
            if error != 0.0 {
                components.push(error);
            }
            let (next, error) = two_sum(product, next);
            if error != 0.0 {
                components.push(error);
            }
            sum = next;
        }
        if sum != 0.0 {
            components.push(sum);
        }
        Self(components)
    }
}
