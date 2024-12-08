use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use rust_hnsw::distances::euclidean;
use rust_hnsw::hnsw::HNSW;

const SEED: u64 = 1234;
const LOWD: usize = 3;
const HIGHD: usize = 784;

fn benchmark_low_d_distance(c: &mut Criterion) {
    c.bench_function("low-d distance", |b| {
        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);

        b.iter_batched(
            || {
                (
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(LOWD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(LOWD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )
            },
            |(x, y): ([f64; LOWD], [f64; LOWD])| euclidean(&x, &y),
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
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(HIGHD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(HIGHD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )
            },
            |(x, y): ([f64; HIGHD], [f64; HIGHD])| euclidean(&x, &y),
            BatchSize::LargeInput,
        );
    });
}

fn benchmark_low_d_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw low-d insertion");

    for size in [1, 100] {
        group.bench_function(format!("{size}"), |b| {
            let rng = SmallRng::seed_from_u64(SEED);
            let mut index = HNSW::new(
                black_box(16),
                black_box(100),
                black_box(euclidean),
                black_box(rng),
            );

            let mut rng_data = SmallRng::seed_from_u64(SEED);
            let data_distribution = Uniform::new(-1.0, 1.0);

            b.iter_batched(
                || {
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(LOWD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                },
                |vector: [f64; LOWD]| index.insert(vector),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_low_d_search(c: &mut Criterion) {
    c.bench_function("low-d search", |b| {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(
            black_box(16),
            black_box(100),
            black_box(euclidean),
            black_box(rng),
        );

        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);
        for _ in 0..100 {
            let vector: [f64; LOWD] = data_distribution
                .sample_iter(&mut rng_data)
                .take(LOWD)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            index.insert(vector);
        }

        b.iter_batched(
            || {
                data_distribution
                    .sample_iter(&mut rng_data)
                    .take(LOWD)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap()
            },
            |query| {
                let _ = index.search(&query, 3);
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
            let mut index = HNSW::new(
                black_box(16),
                black_box(100),
                black_box(euclidean),
                black_box(rng),
            );

            let mut rng_data = SmallRng::seed_from_u64(SEED);
            let data_distribution = Uniform::new(-1.0, 1.0);

            b.iter_batched(
                || {
                    data_distribution
                        .sample_iter(&mut rng_data)
                        .take(HIGHD)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                },
                |vector: [f64; HIGHD]| index.insert(vector),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_high_d_search(c: &mut Criterion) {
    c.bench_function("high-d search", |b| {
        let rng = SmallRng::seed_from_u64(SEED);
        let mut index = HNSW::new(
            black_box(16),
            black_box(100),
            black_box(euclidean),
            black_box(rng),
        );

        let mut rng_data = SmallRng::seed_from_u64(SEED);
        let data_distribution = Uniform::new(-1.0, 1.0);
        for _ in 0..100 {
            let vector: [f64; HIGHD] = data_distribution
                .sample_iter(&mut rng_data)
                .take(HIGHD)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            index.insert(vector);
        }

        b.iter_batched(
            || {
                data_distribution
                    .sample_iter(&mut rng_data)
                    .take(HIGHD)
                    .collect::<Vec<f64>>()
                    .try_into()
                    .unwrap()
            },
            |query| {
                let _ = index.search(&query, 3);
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    benchmark_low_d_distance,
    benchmark_high_d_distance,
    benchmark_low_d_insertion,
    benchmark_low_d_search,
    benchmark_high_d_insertion,
    benchmark_high_d_search,
);
criterion_main!(benches);
