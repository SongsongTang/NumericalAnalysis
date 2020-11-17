import numpy as np
from scipy.linalg import hilbert

H = hilbert(6)
#H = np.array([[2., -1., 3., 6.2], [1.,2., 3. ,7.6],[7.4, 3., 7., 2.], [9., 4., 2., 6.]])
print("H: \n", H)
x = np.array([1, 2, 3, 4, 5, 6])[:, np.newaxis]
#print(x)
b = np.dot(H, x)
print("b: \n", b)
# Now based on H and b, calculate x
class solveEquations(object):
    """
        Solve equations using Gauss method.
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b
        #self.c = np.concatenate((A, b), axis=1)
        #print(self.c)
        self.A_det = 0
        self.A_shape = A.shape
        #print(self.A_shape)
        #self.A_shape[0] = self.A_shape[0]
        #self.A_shape[1] = self.A_shape[1]

    def gauss(self):
        self.A_det = 1
        for k in range(self.A_shape[1] - 1):
            print("按" + str(k+1) + "列选主元")
            a = self.A[k:, k:]
            i_max = np.argmax(np.abs(a[:, 0]))
            if a[i_max, 0] == 0:
                print("det(A) = 0, break!")
                exit(0)
            if i_max == 0:
                pass
            else:
                a[[0, i_max]] = a[[i_max, 0]]
                #print("a", a)
                self.b[[k, i_max+k]] = self.b[[i_max+k, k]]
                self.A_det = -self.A_det
            for i in range(1, a.shape[0]):
                m = np.ones(a.shape[0])
                m[i] = a[i, 0] / a[0, 0]
                a[i] = a[i] - a[0] * m[i]
                self.b[i + k] = self.b[i + k] - self.b[k] * m[i]
                #print("a", a)
                #a[i, 0] = m[i]
                #for j in range(1, a.shape[0]):
                #    a[i, j] = a[i, j] - m[i] * a[0, j]
                #self.b[i] = self.b[i] - m[i] * b[0]
            self.A_det = a[0, 0] * self.A_det
            self.A[k:, k:] = a
            print(self.A)
            print(self.b)
        if self.A[-1, -1] == 0:
            print("det(A) = 0, break!")
            exit(0)

        print("回代...")
        self.b[-1] = self.b[-1] / self.A[-1, -1]
        for i in range(self.A_shape[0] - 2, -1, -1):
            self.b[i] = (self.b[i] - np.sum(self.b[i+1:].T * self.A[i, i+1:])) / self.A[i, i]
            #self.b[i] = (self.b[i] - np.dot(self.A[i, i+1:], self.b[i+1:])) / self.A[i, i]

        self.A_det = self.A[-1, -1] * self.A_det
            #k_indices = np.concatenate((np.arange(k), np.argsort(np.abs(self.A[k:, k]))[::-1] + k))
        print("方程组求解完成！")


if __name__ == "__main__":
    e = solveEquations(H, b)
    e.gauss()
    print(e.b, e.A_det)
