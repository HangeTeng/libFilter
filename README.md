好的，遵命。这里是完整的 `README.md` 内容，全部放在一个代码块中，方便您直接复制粘贴。

```markdown
<p align="center">
  <!-- You can place a project logo or title image here -->
  <!-- e.g., <img src="path/to/your/logo.png" alt="libFilter Logo" width="400"/> -->
  <h1 align="center">libFilter</h1>
</p>

=====

<!-- Add a build status badge from GitHub Actions here -->
<!-- ![Build Status](https://github.com/your-username/libFilter/actions/workflows/build-test.yml/badge.svg) -->

A fast, portable, and modern Python 3 library for advanced probabilistic data structures. The primary design goal of this library is to provide high-quality implementations of both classic and cutting-edge filters, while maintaining a **clean, extensible, and easy-to-use** API.

**Basic Filters (Set Membership & Difference):**
*   **Bloom Filter (BF)**: The classic space-efficient probabilistic data structure for set membership testing.
*   **Counting Bloom Filter (CBF)**: An extension of the Bloom Filter that supports element deletions.
*   **Invertible Bloom Lookup Table (IBLT)** [[GKMMS2011]](https://dl.acm.org/doi/10.1145/2043164.2018449): A powerful data structure that can efficiently recover the symmetric difference between two sets.
*   **Robust IBLT (RIBLT)** [[GMR2011]](https://ieeexplore.ieee.org/document/6197501): A robust version of IBLT based on rate-less coding principles, making it insensitive to parameters and enabling streaming data processing.

**Filters for Secure Aggregation (Privacy-Preserving Aggregation):**
*   **IBLT for Neural Networks (IBLT4NN)**: Utilizes Galois Fields and cryptographic primitives (PRV) to enable the secure aggregation of numerical updates (e.g., gradients, weights), preserving the privacy of individual contributions.
*   **Rate-less IBLT for Neural Networks (RIBLT4NN)**: Combines the streaming capabilities of RIBLT with the secure aggregation properties of IBLT4NN, ideal for scenarios like federated learning.

## Introduction

`libFilter` provides a suite of probabilistic data structures for various applications. It begins with the classic **BF** and **CBF** for high-speed set membership queries. It then advances to the more powerful **IBLT** and **RIBLT**, which can not only test for membership but also efficiently reconstruct the **symmetric difference** between two sets without transferring the sets themselves. This capability is invaluable in network synchronization, database reconciliation, and other related problems.

A core feature of this library is its collection of variants designed for **privacy-preserving machine learning**. The **IBLT4NN** and **RIBLT4NN** implementations leverage finite field arithmetic and a Pseudo-Random Vector (PRV) to securely aggregate numerical updates from multiple parties while protecting individual data privacy. This makes them ideal tools for federated learning, secure multi-party computation, and other privacy-centric domains.

All implementations are built upon an elegant and robust abstraction layer, ensuring a consistent API and high code quality across the library.

## Build and Installation

This is a pure Python library, ensuring cross-platform compatibility. It requires **Python 3.8+**.

The main dependencies are `galois` (for finite field arithmetic) and `pycryptodome` (for cryptographic primitives).

The library can be installed using standard Python packaging tools. We highly recommend using a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/libFilter.git
cd libFilter

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the library in editable mode along with its dependencies
pip install -e .
```
Installing in editable mode (`-e`) is convenient for development, as any changes you make to the source code will be immediately effective without requiring a reinstallation.

## Running Tests

`libFilter` comes with a comprehensive test suite using `pytest`. To run the tests, first install the testing dependencies:
```bash
pip install ".[test]"
```
Then, run `pytest` from the root of the project directory:
```bash
pytest
```
To enable detailed, verbose output of the filter states during test execution for debugging purposes, use the `--verbose-filters` flag:
```bash
pytest -s --verbose-filters
```

## Usage

The API of `libFilter` is designed to be intuitive. Below are a few examples. For more detailed usage, please refer to the files in the `tests/` directory.

**Example 1: IBLT for Set Reconciliation**
```python
from libFilter.core.utils import HashMapping
from libFilter.filters.iblt import IBLT, IBLTItem

# Two sets of items at different clients
set_a = {IBLTItem("user1", "sword"), IBLTItem("common_item", "shield")}
set_b = {IBLTItem("user2", "potion"), IBLTItem("common_item", "shield")}

# Configure and create the IBLT
hash_map = HashMapping.from_seeds(['s1', 's2', 's3'], table_size=50)
iblt = IBLT(hash_map)

# Reconcile the sets: one client pushes, the other removes
for item in set_a:
    iblt.push(item)
for item in set_b:
    iblt.remove(item)

# Decode the IBLT to find the differences
added, removed = iblt.peel()

print(f"Items only in Set A: {added}")
# Expected: {IBLTItem(key='user1', value='sword')}
print(f"Items only in Set B: {removed}")
# Expected: {IBLTItem(key='user2', value='potion')}
```

**Example 2: RIBLT4NN for Secure Aggregation**
```python
import math
from libFilter.core.prv import PRV
from libFilter.filters4nn.riblt4nn import RIBLT4NN, NNItem

# 1. Set up a PRV for secure index encoding
prv = PRV(n=2048) # Assume 2048 possible weight indices

# 2. Create a streaming RIBLT4NN
riblt = RIBLT4NN(prv)

# 3. Process streaming updates from multiple clients
# Client 1's updates arrive
riblt.push(NNItem(idx=20, weight=0.8))
riblt.push(NNItem(idx=1500, weight=-0.3))

# The table can be expanded at any time
riblt.expand(100)

# Client 2's updates arrive
riblt.push(NNItem(idx=20, weight=0.1))
riblt.push(NNItem(idx=200, weight=0.5))

# 4. Finalize the stream and decode
riblt.set_done_expanding()
while riblt.peel():
    pass

# 5. Retrieve the final aggregated results
aggregated_weights = riblt.decoded_weights
print(f"Aggregated weights: {aggregated_weights}")

# Expected output: {20: 0.9, 1500: -0.3, 200: 0.5} (or close floating-point values)
assert math.isclose(aggregated_weights[20], 0.9)
```

## Citing `libFilter`

If you use `libFilter` in your research or project, please consider citing it.

```bibtex
@misc{libFilter,
    author = {Your Name},
    title = {{libFilter: An Advanced Probabilistic Data Structure Library in Python}},
    howpublished = {\url{https://github.com/your-username/libFilter}},
    year = {2024}
}
```

## References

[GKMMS2011] - Michael T. Goodrich, Michael Mitzenmacher, Justin Thaler, _The Power of Choice in Data-Streaming Problems_.

[GMR2011] - Michael T. Goodrich, Michael Mitzenmacher, and Rasmus Pagh, _Invertible Bloom lookup tables_.

(You can continue to list other relevant papers for the algorithms implemented.)
```