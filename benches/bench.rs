use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use pprof::criterion::{Output, PProfProfiler};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::SmallRng,
    {Rng, SeedableRng},
};
use rust_hnsw::{distances::euclidean, hnsw::HNSW};
use std::time::Duration;

const SEED: u64 = 1234;
const LOWD: usize = 3;
const HIGHD: usize = 784;

fn sample_vector<const D: usize, R: Rng>(
    distribution: impl Distribution<f64>,
    rng: &mut R,
) -> [f64; D] {
    distribution
        .sample_iter(rng)
        .take(D)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn get_config() -> Criterion {
    Criterion::default()
        .significance_level(0.1)
        .sample_size(100)
        .measurement_time(Duration::new(10, 0))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
}

fn benchmark_low_d_distance(c: &mut Criterion) {
    c.bench_function("low-d distance", |b| {
        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);

        b.iter_batched(
            || {
                (
                    sample_vector(data_distribution, &mut rng_data),
                    sample_vector(data_distribution, &mut rng_data),
                )
            },
            |(x, y): ([_; LOWD], [_; LOWD])| euclidean(black_box(&x), black_box(&y)),
            BatchSize::SmallInput,
        );
    });
}

fn benchmark_high_d_distance(c: &mut Criterion) {
    c.bench_function("high-d distance", |b| {
        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);

        b.iter_batched(
            || {
                (
                    sample_vector(data_distribution, &mut rng_data),
                    sample_vector(data_distribution, &mut rng_data),
                )
            },
            |(x, y): ([_; HIGHD], [_; HIGHD])| euclidean(black_box(&x), black_box(&y)),
            BatchSize::LargeInput,
        );
    });
}

fn benchmark_low_d_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw low-d insertion");

    for size in [1, 100] {
        group.bench_function(format!("{size}"), |b| {
            let rng = SmallRng::seed_from_u64(SEED);
            let mut index = HNSW::new(16, 100, euclidean, rng);

            let mut rng_data = SmallRng::seed_from_u64(SEED);
            let data_distribution = Uniform::new(-1.0, 1.0);

            b.iter_batched(
                || {
                    (0..size)
                        .map(|_| sample_vector(data_distribution, &mut rng_data))
                        .collect()
                },
                |vectors: Vec<[_; LOWD]>| {
                    vectors.iter().for_each(|&v| index.insert(black_box(&v)));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_low_d_search(c: &mut Criterion) {
    c.bench_function("low-d search", |b| {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(16, 100, euclidean, rng);

        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);
        for _ in 0..100 {
            let vector: [_; LOWD] = sample_vector(data_distribution, &mut rng_data);
            index.insert(&vector);
        }

        b.iter_batched(
            || sample_vector(data_distribution, &mut rng_data),
            |query| {
                let _ = index.search(black_box(&query), black_box(3));
            },
            BatchSize::SmallInput,
        );
    });
}

fn benchmark_high_d_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw high-d insertion");

    for size in [1, 100] {
        group.bench_function(format!("{size}"), |b| {
            let rng = SmallRng::seed_from_u64(SEED);
            let mut index = HNSW::new(16, 100, euclidean, rng);

            let mut rng_data = SmallRng::seed_from_u64(SEED);
            let data_distribution = Uniform::new(-1.0, 1.0);

            b.iter_batched(
                || {
                    (0..size)
                        .map(|_| sample_vector(data_distribution, &mut rng_data))
                        .collect()
                },
                |vectors: Vec<[_; HIGHD]>| {
                    vectors.iter().for_each(|&v| index.insert(black_box(&v)));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_high_d_search(c: &mut Criterion) {
    c.bench_function("high-d search", |b| {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(16, 100, euclidean, rng);

        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);
        for _ in 0..100 {
            let vector: [_; HIGHD] = sample_vector(data_distribution, &mut rng_data);
            index.insert(&vector);
        }

        b.iter_batched(
            || sample_vector(data_distribution, &mut rng_data),
            |query| {
                let _ = index.search(black_box(&query), black_box(3));
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    name = benches;
    config = get_config();
    targets = benchmark_low_d_distance,
    benchmark_high_d_distance,
    benchmark_low_d_insertion,
    benchmark_low_d_search,
    benchmark_high_d_insertion,
    benchmark_high_d_search,
);
criterion_main!(benches);
