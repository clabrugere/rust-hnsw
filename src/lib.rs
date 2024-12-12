pub mod distances;
pub mod hnsw;

#[cfg(test)]
mod tests {
    use super::{distances::euclidean, hnsw::HNSW};
    use rand::{rngs::SmallRng, SeedableRng};

    const SEED: u64 = 1234;

    #[test]
    fn test_new() {
        let rng = SmallRng::seed_from_u64(SEED);
        let index: HNSW<f64, 3, _, _> = HNSW::new(1, 1, euclidean, rng);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.num_levels(), 0);
    }

    #[test]
    fn test_insert() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);

        let vector1 = [1., 2., 3.];
        let vector2 = [4., 5., 6.];
        let vector3 = [7., 8., 9.];

        index.insert(&vector1);
        index.insert(&vector2);
        index.insert(&vector3);

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
        assert!(index.nodes.values().any(|v| v == &vector1));
        assert!(index.nodes.values().any(|v| v == &vector2));
        assert!(index.nodes.values().any(|v| v == &vector3));
    }

    #[test]
    fn test_insert_iterator() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);
        let iterator = (0..3).map(|i| [i as f64; 2]);

        index.insert_batch(iterator);

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_level_density_decay() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);

        index.insert_batch((0..10).map(|i| [i as f64; 2]));

        // check that the number of nodes in levels is smaller the higher the level
        let structure_ok = index.levels.windows(2).all(|w| {
            let (layer_0, layer_1) = (&w[0], &w[1]);
            layer_0.len() >= layer_1.len()
        });

        assert!(structure_ok);
    }

    #[test]
    fn test_max_connections() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);

        index.insert_batch((0..10).map(|i| [i as f64; 2]));

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
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);
        let vector = [1., 2., 3.];

        assert!(index.search(&vector, 1).is_err());
    }

    #[test]
    fn test_search_exact() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);
        let vector = [1., 2., 3.];

        index.insert(&vector);
        let result = index.search(&vector, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, &vector);
        assert!(result[0].distance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_search() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);

        let vector1 = [1., 2., 3.];
        let vector2 = [0., 0., 0.];
        let vector3 = [10., 20., 30.];

        index.insert(&vector1);
        index.insert(&vector2);
        index.insert(&vector3);

        let query = [1.1, 2.1, 3.1];
        let result = index.search(&query, 3).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(
            result.iter().map(|r| r.vector).collect::<Vec<_>>(),
            &[&vector1, &vector2, &vector3]
        );
    }

    #[test]
    fn test_clear() {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(8, 8, euclidean, rng);

        index.insert_batch((0..10).map(|i| [i as f64; 2]));

        assert_eq!(index.len(), 10);

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }
}
