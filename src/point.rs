use crate::PointInterface;

#[derive(Clone, Debug)]
pub struct Point<const D: usize>(pub Vec<f32>);

impl<const D: usize> PointInterface<D> for Point<D> {
    fn distance(&self, other: &Self) -> f32 {
        -cosine_similarity(&self, &other) + 1.0
    }

    fn add(&self, other: &Self) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .zip(other.to_f32_vec())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }
    fn div(&self, divisor: &usize) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .map(|v| v / *divisor as f32)
                .collect(),
        )
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().copied().collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().collect())
    }
}

fn dot_product<const D: usize>(vec1: &Point<D>, vec2: &Point<D>) -> f32 {
    assert_eq!(vec1.0.len(), vec2.0.len());
    let dim: usize = vec1.0.len();
    let mut result = 0.0;
    for i in 0..dim {
        result += vec1.0[i] * vec2.0[i];
    }
    result
}

fn norm<const D: usize>(vec: &Point<D>) -> f32 {
    let dim = vec.0.len();
    let mut result = 0.0;
    for i in 0..dim {
        result += vec.0[i] * vec.0[i];
    }
    result.sqrt()
}

fn cosine_similarity<const D: usize>(vec1: &Point<D>, vec2: &Point<D>) -> f32 {
    let dot = dot_product(vec1, vec2);
    let norm1 = norm(vec1);
    let norm2 = norm(vec2);

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot / (norm1 * norm2)
}
