pub mod distances;
mod errors;
pub mod hnsw;

#[cfg(test)]
mod tests {
    use super::{distances::euclidean, hnsw::HNSW};
    use rand::{rngs::SmallRng, SeedableRng};

    const SEED: u64 = 1234;

    fn create_index() -> HNSW<f64, 3, for<'a, 'b> fn(&'a [f64], &'b [f64]) -> f64, SmallRng> {
        let rng = SmallRng::seed_from_u64(SEED);
        HNSW::new(8, 8, euclidean, rng)
    }

    #[test]
    fn test_new() {
        let index = create_index();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.num_levels(), 0);
    }

    #[test]
    fn test_insert() {
        let mut index = create_index();

        let vector1 = [1., 2., 3.];
        let vector2 = [4., 5., 6.];
        let vector3 = [7., 8., 9.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
        assert!(index.nodes.values().any(|v| v == &vector1));
        assert!(index.nodes.values().any(|v| v == &vector2));
        assert!(index.nodes.values().any(|v| v == &vector3));
    }

    #[test]
    fn test_insert_iterator() {
        let mut index = create_index();
        let iterator = (0..3).map(|i| [i as f64; 3]);

        index.insert_batch(iterator).unwrap();

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_level_density_decay() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        // check that the number of nodes in levels is smaller the higher the level
        let structure_ok = index.levels.windows(2).all(|w| {
            let (layer_0, layer_1) = (&w[0], &w[1]);
            layer_0.len() >= layer_1.len()
        });

        assert!(structure_ok);
    }

    #[test]
    fn test_max_connections() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        let structure_ok = index.levels.iter().enumerate().all(|(level_index, level)| {
            level.values().all(move |edges| {
                let max_connections = if level_index > 0 {
                    index.max_connections
                } else {
                    index.max_connections_0
                };
                edges.len() <= max_connections
            })
        });

        assert!(structure_ok);
    }

    #[test]
    fn test_search_empty() {
        let mut index = create_index();
        let vector = [1., 2., 3.];

        assert!(index.search(&vector, 1).is_err());
    }

    #[test]
    fn test_search_exact() {
        let mut index = create_index();
        let vector = [1., 2., 3.];

        index.insert(&vector).unwrap();
        let result = index.search(&vector, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, &vector);
        assert!(result[0].distance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_search_ordering() {
        let mut index = create_index();
        let vector1 = [1., 1., 1.]; // distance ~1.73 from origin
        let vector2 = [0., 0., 0.]; // distance 0 from origin
        let vector3 = [2., 2., 2.]; // distance ~3.46 from origin
        let vector4 = [0.5, 0.5, 0.5]; // distance ~0.87 from origin

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();
        index.insert(&vector4).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 4).unwrap();

        assert_eq!(result.len(), 4);
        // Results should be ordered by distance (closest first)
        assert_eq!(result[0].vector, &vector2); // closest
        assert_eq!(result[1].vector, &vector4); // second closest
        assert_eq!(result[2].vector, &vector1); // third closest
        assert_eq!(result[3].vector, &vector3); // farthest

        // Verify distances are in ascending order
        for i in 1..result.len() {
            assert!(result[i - 1].distance <= result[i].distance);
        }
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let mut index = create_index();
        let vector1 = [1., 2., 3.];
        let vector2 = [4., 5., 6.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 10).unwrap(); // k > index size

        assert_eq!(result.len(), 2); // Should return all available vectors
        assert!(result.iter().any(|r| r.vector == &vector1));
        assert!(result.iter().any(|r| r.vector == &vector2));
    }

    #[test]
    fn test_search_k_zero() {
        let mut index = create_index();
        let vector = [1., 2., 3.];
        index.insert(&vector).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 0).unwrap();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_search_with_duplicates() {
        let mut index = create_index();
        let vector1 = [1., 2., 3.];
        let vector2 = [1., 2., 3.]; // duplicate
        let vector3 = [4., 5., 6.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();

        let query = [1., 2., 3.];
        let result = index.search(&query, 3).unwrap();

        assert_eq!(result.len(), 3);
        // First two results should have distance 0 (exact matches)
        assert!(result[0].distance.abs() < f64::EPSILON);
        assert!(result[1].distance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        assert_eq!(index.len(), 10);

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }
}
