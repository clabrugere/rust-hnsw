use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

/// Compute the squared L2 distance between two vectors
pub fn squared_euclidean<T>(x: &[T], y: &[T]) -> T
where
    T: Sized + Copy + Sub<Output = T> + Mul<Output = T> + Sum,
{
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| (xi - yi) * (xi - yi))
        .sum()
}

/// Compute the L2 distance between two vectors
pub fn euclidean<T>(x: &[T], y: &[T]) -> f64
where
    T: Sized + Copy + Sub<Output = T> + Mul<Output = T> + Sum + Into<f64>,
{
    squared_euclidean(x, y).into().sqrt()
}

/// Compute the cosine distance between two vectors and return a f64
pub fn cosine<T>(x: &[T], y: &[T]) -> f64
where
    T: Sized + Copy + Into<f64>,
{
    let (mut x_norm, mut y_norm, mut dot) = (0.0, 0.0, 0.0);

    for (&xi, &yi) in x.iter().zip(y) {
        let xi_f64: f64 = xi.into();
        let yi_f64: f64 = yi.into();
        x_norm += xi_f64 * xi_f64;
        y_norm += yi_f64 * yi_f64;
        dot += xi_f64 * yi_f64;
    }

    1.0 - dot / (x_norm.sqrt() * y_norm.sqrt())
}

pub fn manhattan<T>(x: &[T], y: &[T]) -> f64
where
    T: Sized + Copy + Sub<Output = T> + Into<f64>,
{
    x.iter()
        .zip(y)
        .map(|(&xi, &yi)| (xi - yi).into().abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_zero_distance() {
        let x = [1, 2, 3];
        let y = [1, 2, 3];
        assert_eq!(squared_euclidean(&x, &y), 0);
    }

    #[test]
    fn test_squared_euclidean_zero_values() {
        let x = [0, 0, 0];
        let y = [0, 0, 0];
        assert_eq!(squared_euclidean(&x, &y), 0);
    }

    #[test]
    fn test_squared_euclidean_integers() {
        let x = [1, -2, 3];
        let y = [-1, 2, -3];
        assert_eq!(squared_euclidean(&x, &y), 56);
    }

    #[test]
    fn test_squared_euclidean_floating_point() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let expected: f64 = squared_euclidean(&x, &y) - 27.0;
        assert!(expected.abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert!((cosine(&x, &y) - 1.0) < f64::EPSILON);
    }

    #[test]
    fn test_cosine_zero_distance() {
        let x = [1.0, 2.0, 3.0];
        assert!(cosine(&x, &x).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, -2.0, -3.0];
        assert!((cosine(&x, &y) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let x = [0.0, 0.0, 0.0];
        let y = [1.0, 2.0, 3.0];
        assert!(cosine(&x, &y).is_nan());
    }

    #[test]
    fn test_cosine() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let expected = 0.025368153802923787;
        assert!((cosine(&x, &y) - expected).abs() < 1e-16);
    }

    #[test]
    fn test_manhattan() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];
        let d = manhattan(&x, &y);
        assert!((d - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_manhattan_zero_distance() {
        let x = [5.0, -3.2, 0.0];
        let y = [5.0, -3.2, 0.0];
        let d = manhattan(&x, &y);
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_manhattan_negative_values() {
        let x = [-1.0, -2.0, -3.0];
        let y = [1.0, 2.0, 3.0];
        let d = manhattan(&x, &y);
        assert!((d - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_manhattan_integer_input() {
        let x = [1, 2, 3, 4];
        let y = [4, 3, 2, 1];
        let d = manhattan(&x, &y);
        assert!((d - 8.0).abs() < 1e-9);
    }
}
