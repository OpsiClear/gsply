# GSData Container Type Performance Benchmark Results

Benchmark comparing regular dataclass vs frozen dataclass vs NamedTuple for the GSData container.

## Test Configuration
- **Data Size**: 100,000 Gaussians
- **Platform**: Windows
- **Test Data**: Full Gaussian splatting data with SH degree 3

## Results Summary

### 1. Container Creation Time

| Container Type | Total Time (ms) | Avg Time (ns) | Relative Speed |
|----------------|-----------------|---------------|----------------|
| **Regular Dataclass (current)** | **1.982** | **198.2** | **1.00x (baseline)** |
| Frozen Dataclass | 6.649 | 664.9 | 0.30x (3.4x slower) |
| NamedTuple | 2.642 | 264.2 | 0.75x (1.3x slower) |

**Winner: Regular Dataclass** - Fastest creation by significant margin

### 2. Attribute Access Performance

| Container Type | Total Time (ms) | Avg Time (ns) | Relative Speed |
|----------------|-----------------|---------------|----------------|
| **Regular Dataclass (current)** | **49.211** | **49.2** | **1.00x (baseline)** |
| Frozen Dataclass | 68.643 | 68.6 | 0.72x (1.4x slower) |
| NamedTuple | 87.332 | 87.3 | 0.56x (1.8x slower) |

**Winner: Regular Dataclass** - Fastest attribute access

### 3. Indexing/Unpacking Performance

| Container Type | Total Time (ms) | Avg Time (ns) |
|----------------|-----------------|---------------|
| **Frozen Dataclass** | **3.010** | **30.1** |
| Regular Dataclass | 3.143 | 31.4 |
| NamedTuple | 10.432 | 104.3 |

**Winner: Frozen Dataclass** - Slightly faster indexing (marginal difference)

### 4. Memory Overhead

| Container Type | Container Size (bytes) | Array Data (MB) |
|----------------|------------------------|-----------------|
| Regular Dataclass | 48 | 47.20 |
| Frozen Dataclass | 48 | 47.20 |
| NamedTuple | 96 | 47.20 |

**Winner: Regular/Frozen Dataclass** - Both have minimal overhead (NamedTuple uses 2x container memory)

## Analysis

### Regular Dataclass (Current Implementation)
**Pros:**
- Fastest creation time (3.4x faster than frozen)
- Fastest attribute access
- Minimal memory overhead (48 bytes)
- Mutable (allows modifications if needed)
- Good balance of performance and usability

**Cons:**
- Mutable (could be modified accidentally, though this is rarely an issue)

### Frozen Dataclass
**Pros:**
- Immutable (prevents accidental modification)
- Nearly identical memory overhead to regular dataclass
- Slightly faster indexing

**Cons:**
- 3.4x slower creation time
- 1.4x slower attribute access
- Cannot modify after creation (may limit some use cases)

### NamedTuple
**Pros:**
- Immutable
- Supports indexing and unpacking natively
- Lightweight for small objects

**Cons:**
- 1.3x slower creation
- 1.8x slower attribute access
- 3.5x slower indexing than dataclasses
- 2x memory overhead (96 vs 48 bytes)

## Recommendation

**Keep the current Regular Dataclass implementation.**

### Reasons:
1. **Best Overall Performance**: Fastest creation (3.4x) and access (1.4x-1.8x)
2. **Minimal Overhead**: Only 48 bytes container overhead
3. **Read-Heavy Workload**: GSData is primarily read from files and accessed, making attribute access speed critical
4. **Creation Matters**: Files are read frequently, so creation speed impacts I/O performance
5. **Negligible Mutability Risk**: In practice, users rarely modify GSData after reading

### Performance Impact on Real Workloads:
- Reading 100K Gaussians: Regular dataclass creates container in ~200ns vs ~665ns for frozen
- Accessing data 1M times: Regular dataclass takes 49ms vs 69ms for frozen, 87ms for NamedTuple
- For typical workflows with many reads: **Regular dataclass provides 30-80% better performance**

## Conclusion

The current implementation using a regular dataclass is optimal for gsply's use case. The performance advantages in both creation and access far outweigh the theoretical safety benefits of immutability, especially given that:
- NumPy arrays themselves are mutable regardless of container type
- The API design encourages reading data, not modifying containers
- The performance difference is significant in read-heavy workloads

**No changes recommended.**
