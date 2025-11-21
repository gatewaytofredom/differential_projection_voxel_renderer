# Voxel Software Rasterizer

A high-performance voxel rendering engine built in Rust, implementing a fully software-based rasterization pipeline optimized for modern multi-core CPUs. This project showcases advanced optimization techniques including SIMD vectorization, parallel rendering, and cache-efficient data structures.

## Features

### Core Rendering
- **Software Rasterization**: Custom scanline rasterizer with depth buffering
- **Parallel Rendering**: Multi-core stripe-based rasterization using Rayon (6-8x speedup)
- **Binary Greedy Meshing**: Optimized voxel-to-mesh conversion using bitwise operations
- **Frustum & Horizon Culling**: Intelligent chunk culling for large view distances
- **Texture Mapping**: UV-mapped block textures with bilinear filtering

### Performance Optimizations
- **SIMD Acceleration**: AVX2-optimized vertex transformation and projection (2-3x speedup)
- **Memory Bandwidth**: Compressed vertex format (50% reduction) and optimized layouts
- **Cache Efficiency**: Data-oriented design with minimal pointer chasing
- **Allocation-Free Hot Paths**: Workspace reuse eliminates 192+ allocations per chunk
- **Differential Projection**: Exploit voxel geometry for 8x fewer matrix operations

### Developer Features
- **Comprehensive Benchmarks**: Hardware performance counters, cache miss tracking, profiling
- **Modular Architecture**: Each component independently testable and benchmarkable
- **FPS Camera Controls**: Mouse look and WASD movement
- **Performance Logging**: Per-frame timing of all rendering stages

## Performance Characteristics

Optimized for **Intel i5-12400** (6 cores, AVX2, 48KB L1 / 1.25MB L2 / 18MB L3):

### Frame Times (1280x720, view distance 12)
- **60 FPS** (16.67ms target): Easily achieved
- **Chunk Meshing**: <1ms per 32³ chunk (cached: 0ms)
- **Rasterization**: 5-15ms (70-90% of frame time, parallelized 6-8x)
- **Culling**: ~0.3ms (frustum + horizon)
- **Memory Usage**: ~15MB peak (8MB framebuffer + chunk data)

### Scalability
- **2 cores**: 1.9-2.0x speedup (95-100% efficiency)
- **4 cores**: 3.7-3.9x speedup (92-98% efficiency)
- **8 cores**: 7.0-7.5x speedup (88-94% efficiency)

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed implementation notes and optimization history.

## Controls

- **W/A/S/D** - Move camera forward/left/backward/right
- **Space** - Move up
- **Left Shift** - Move down
- **Mouse** - Look around (click to capture mouse)
- **ESC** - Release mouse or exit

## Building and Running

### Prerequisites

- Rust 1.70+ (edition 2021)
- Linux with X11/Wayland support

### Build

```bash
# Debug build
cargo build

# Release build (with optimizations)
cargo build --release

# Run
cargo run --release
```

### Benchmarks

Comprehensive benchmark suite with hardware performance counter integration:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suites
cargo bench --bench meshing          # Binary greedy meshing
cargo bench --bench rendering        # Rasterization pipeline
cargo bench --bench microbench       # Hot-path profiling
cargo bench --bench differential_projection  # SIMD projection

# With function call profiling
cargo bench --features profiling

# Hardware profiling (Linux only)
perf stat -d cargo bench
perf stat -e cache-misses,cache-references cargo bench --bench rendering
```

See [BENCHMARKING.md](BENCHMARKING.md) for the complete profiling guide including:
- Cache miss analysis
- Branch prediction metrics
- IPC (instructions per cycle) analysis
- Hardware counter integration

## Key Optimizations

This engine demonstrates several advanced optimization techniques:

### 1. SIMD Vectorization (AVX2)
- **Vertex Transformation**: Process 8 vertices simultaneously (~1.3 ns/vertex)
- **Differential Projection**: Exploit axis-aligned geometry (2.37x speedup, 8x fewer FMAs)
- **Batch Processing**: Structure-of-arrays layout for efficient vectorization

### 2. Parallel Rendering
- **Stripe-Based**: Horizontal framebuffer bands, one per core
- **Work-Stealing**: Rayon automatically balances load across threads
- **Zero Contention**: Thread-local rasterizers, disjoint memory regions

### 3. Memory Bandwidth Optimizations
- **Vertex Compression**: u8 local coordinates (50% reduction: 16→8 bytes)
- **Quad Compression**: u8 fields (75% reduction: 16→4 bytes)
- **Allocation-Free**: Workspace reuse eliminates ~192+ allocations per chunk
- **Cache-Friendly**: Flat arrays, aligned data structures, sequential access patterns

### 4. Culling Pipeline
- **Frustum Culling**: 6-plane AABB tests eliminate off-screen chunks
- **Horizon Culling**: Distance-based occlusion for terrain (20-30% reduction)
- **Backface Culling**: Normal-based triangle rejection (29% reduction)
- **Sub-pixel Culling**: Ultra-conservative (0.05px threshold) for degenerate triangles

### 5. Data-Oriented Design
- **Binary Greedy Meshing**: Bitwise operations process 32 voxels/iteration
- **Mesh Caching**: Immutable chunks reuse previously generated meshes
- **Direct Array Access**: Branchless neighbor checks (eliminates ~7,000 branches/chunk)

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete implementation details.

## Testing

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test meshing
cargo test rendering
cargo test horizon_culling

# Release mode tests (recommended for performance validation)
cargo test --release
```

**Test Coverage**: 75+ tests covering:
- Binary greedy meshing algorithm
- Software rasterization pipeline
- Frustum and horizon culling
- SIMD vertex transformation
- Differential projection
- Rendering correctness

## Project Structure

```
src/
├── main.rs              # Main render loop, culling, binning
├── voxel/               # Chunk data structures, terrain generation
├── meshing/             # Binary greedy meshing algorithm
├── rendering/           # Rasterizer, framebuffer, SIMD transforms
│   ├── rasterizer.rs
│   ├── simd_vertex.rs
│   ├── differential_projection.rs
│   └── ...
├── camera/              # FPS camera, frustum extraction
└── perf/                # Performance profiling infrastructure

benches/                 # Criterion benchmarks
tests/                   # Integration tests
reference_material/      # Binary greedy meshing reference implementation
```

## References

- [Binary Greedy Meshing](https://github.com/cgerikj/binary-greedy-meshing) - Core meshing algorithm
- [Rust Optimization Guide](reference_material/Rust%20Optimization.md) - Performance best practices
- [BENCHMARKING.md](BENCHMARKING.md) - Complete profiling guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Implementation details and design decisions

## License

MIT License - See LICENSE file for details
