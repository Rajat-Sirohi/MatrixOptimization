"""
Feel free to add more test classes and/or tests that are not provided by the skeleton code!
Make sure you follow these naming conventions: https://docs.pytest.org/en/reorganize-docs/goodpractices.html#test-discovery
for your new tests/classes/python files or else they might be skipped.
"""
from utils import *

"""
For each operation, you should write tests to test correctness on matrices of different sizes.
Hint: use rand_dp_nc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""

class TestMulCorrectness:
    def test_mul(self):
        dp_m1, nc_m1 = rand_dp_nc_matrix([[1,2],[3,4]]);
        dp_m2, nc_m2 = rand_dp_nc_matrix([[1,1,1],[0,0,0]]);

        nc_m = nc_m1 * nc_m2;
        dp_m = dp_m1 * dp_m2;

        print("Correct:", cmp_dp_nc_matrix(dp_m1, nc_m1));
        print("Correct:", cmp_dp_nc_matrix(dp_m2, nc_m2));
        print("Correct:", cmp_dp_nc_matrix(dp_m, nc_m));
        pass

class TestSliceCorrectness:
    def test_slice_after_matrix(self):                           # This doesn't actually represent the AG test
        dp_m1, nc_m1 = rand_dp_nc_matrix([[1,2],[3,4]]);
        dp_m2 = dp_m1[0];
        nc_m2 = nc_m1[0];
        del dp_m1, nc_m1

        # print("Correct:", cmp_dp_nc_matrix(dp_m1, nc_m1));
        print("Correct:", cmp_dp_nc_matrix(dp_m2, nc_m2));
        # print("Correct:", cmp_dp_nc_matrix(dp_m3, nc_m3));
        pass