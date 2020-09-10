#### 当两个规模相当的矩阵做内积时，选择CSC或CSR并没有太大差别，时间效果相当。但是当为***一大一小矩阵***时，就有一些技巧，可以节约时间。
假设B为大矩阵，S为小矩阵:
 - 当CSR格式时，S×B速度较快，与B×S相比节约了一半时间。
 - 当CSC格式时，B×S速度较快，与S×B相比节约一半时间。

```python
import scipy.sparse as sp

def is_csr_instance(mtx):
    if isinstance(mtx, sp.csr_matrix):
        return True
    else:
        return False
    
def is_csc_instance(mtx):
    if isinstance(mtx, sp.csc_matrix):
        return True
    else:
        return False

a_mtx = sp.csc_matrix([[1., 1., 3.]*120])
mtx = sp.csc_matrix([[1., 0., 0.]*120]*30000)

is_csc_instance(a_mtx), is_csc_instance(mtx)

mtx.shape, a_mtx.shape

mtx_T = mtx.T
mtx_T = mtx_T.tocsc()

print is_csc_instance(mtx_T), is_csr_instance(mtx_T)

print u"\n\ncsc little×big"
print type(a_mtx), type(mtx_T)
print a_mtx.shape, mtx_T.shape
%timeit c = a_mtx.dot(mtx_T)

print u"\n\ncsr little×big"
a_mtx_r = a_mtx.tocsr()
mtx_T_r = mtx_T.tocsr()
print type(a_mtx_r), type(mtx_T_r)
print a_mtx_r.shape, mtx_T_r.shape
%timeit c = a_mtx_r.dot(mtx_T_r)

a_mtx_T = a_mtx.T
a_mtx_T = a_mtx_T.tocsc()
mtx_T.shape, a_mtx_T.shape

print "\n\ncsc big×little"
print type(mtx), type(a_mtx_T)
print mtx.shape, a_mtx_T.shape
%timeit c = mtx.dot(a_mtx_T)

print "\n\ncsr big×little"
mtx = mtx.tocsr()
a_mtx_T = a_mtx_T.tocsr()
print type(mtx), type(a_mtx_T)
print mtx.shape, a_mtx_T.shape
%timeit c = mtx.dot(a_mtx_T)

```
