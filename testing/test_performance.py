"""
Feel free to add more test classes and/or tests that are not provided by the skeleton code!
Make sure you follow these naming conventions: https://docs.pytest.org/en/reorganize-docs/goodpractices.html#test-discovery
for your new tests/classes/python files or else they might be skipped.
"""
from utils import *
import time

"""
- We will test you on your performance on add, sub, abs, neg, mul, and pow, so make sure to test these
yourself! We will also test your implementation on matrices on different sizes. It is normal if
your smaller matrices are about the same speed as the naive implementation or even slower.
- Use time.time(), NOT time.perf_counter() to time your program!
- DO NOT count the time for initializing matrices into your performance time. Only count the
time the part where operations are carried out.
- Please also check for correctness while testing for performance!
- We provide the structure for test_small_add. All other tests should have similar structures
"""
class TestSimplePerformance:
    def test_add_sub(self):
        # Initialize matrices using rand_dp_nc_matrix
        dp_m1, nc_m1 = rand_dp_nc_matrix(10000,10000,1);
        dp_m2, nc_m2 = rand_dp_nc_matrix(10000,10000,1);

        nc_start = time.time()
        nc_m = nc_m1 + nc_m2;
        nc_end = time.time()

        dp_start = time.time()
        dp_m = dp_m1 + dp_m2;
        dp_end = time.time()

        # Check for correctness using cmp_dp_nc_matrix and calculate speedup
        print("Correct:", cmp_dp_nc_matrix(dp_m, nc_m));
        print("Ratio =", (dp_end - dp_start) / (nc_end - nc_start))
        pass

    def test_neg_abs(self):
        # Initialize matrices using rand_dp_nc_matrix
        dp_m1, nc_m1 = rand_dp_nc_matrix(10000,10000,1);

        nc_start = time.time()
        nc_m = abs(nc_m1)
        nc_end = time.time()

        dp_start = time.time()
        dp_m = abs(dp_m1)
        dp_end = time.time()

        # Check for correctness using cmp_dp_nc_matrix and calculate speedup
        print("Correct:", cmp_dp_nc_matrix(dp_m, nc_m));
        print("Ratio =", (dp_end - dp_start) / (nc_end - nc_start))
        pass

class TestMulPerformance:
    def test_mul(self):
        # Initialize matrices using rand_dp_nc_matrix
        dp_m1, nc_m1 = rand_dp_nc_matrix(1000,1001,rand=True,low=0,high=1000,seed=0);
        dp_m2, nc_m2 = rand_dp_nc_matrix(1001,1002,rand=True,low=0,high=1000,seed=1);

        nc_start = time.time()
        nc_m = nc_m1 * nc_m2;
        nc_end = time.time()

        dp_start = time.time()
        dp_m = dp_m1 * dp_m2;
        dp_end = time.time()

        # Check for correctness using cmp_dp_nc_matrix and calculate speedup
        print("Correct:", cmp_dp_nc_matrix(dp_m, nc_m));
        print("Ratio =", (dp_end - dp_start) / (nc_end - nc_start))
        pass

    def test_power(self):
        # Initialize matrices using rand_dp_nc_matrix
        dp_m1, nc_m1 = rand_dp_nc_matrix(500,500,rand=True,low=0,high=1000,seed=7);
        power = 10

        nc_start = time.time()
        nc_m = nc_m1 ** power
        nc_end = time.time()

        dp_start = time.time()
        dp_m = dp_m1 ** power
        dp_end = time.time()

        # Check for correctness using cmp_dp_nc_matrix and calculate speedup
        print("Correct:", cmp_dp_nc_matrix(dp_m, nc_m));
        print("Ratio =", (dp_end - dp_start) / (nc_end - nc_start))
        pass

# alias m='cd ..; make; cd testing'; alias t='pytest test_performance.py::TestMulPerformance::test_mul -s'; alias tt='pytest test_performance.py::TestMulPerformance::test_power -s'