# square matrices with addition, subtraction, multiplication
class Matrix:
    def __init__(self, size, array):
        self.size = size
        self.array = array
    def __add__(self, other):
        array = [[0]*self.size for i in xrange(self.size)]
        for i in xrange(self.size):
            for j in xrange(self.size):
                array[i][j] = self.array[i][j]+other.array[i][j]
        return Matrix(self.size, array)
    def __sub__(self, other):
        array = [[0]*self.size for i in xrange(self.size)]
        for i in xrange(self.size):
            for j in xrange(self.size):
                array[i][j] = self.array[i][j]-other.array[i][j]
        return Matrix(self.size, array)
    def __mul__(self, other):
        array = [[0]*self.size for i in xrange(self.size)]
        for i in xrange(self.size):
            for j in xrange(self.size):
                for k in xrange(self.size):
                    array[i][j] += self.array[i][k]*other.array[k][j]
        return Matrix(self.size, array)
    def __str__(self):
        rows = [' '.join([str(e) for e in r]) for r in self.array]
        return '\n'.join(rows)
    def ul(self):
        array = [[0]*(self.size/2) for i in xrange(self.size/2)]
        for i in xrange(self.size/2):
            for j in xrange(self.size/2):
                array[i][j] = self.array[i][j]
        return Matrix(self.size/2, array)
    def ur(self):
        array = [[0]*(self.size/2) for i in xrange(self.size/2)]
        for i in xrange(self.size/2):
            for j in xrange(self.size/2):
                array[i][j] = self.array[i][j+self.size/2]
        return Matrix(self.size/2, array)
    def ll(self):
        array = [[0]*(self.size/2) for i in xrange(self.size/2)]
        for i in xrange(self.size/2):
            for j in xrange(self.size/2):
                array[i][j] = self.array[i+self.size/2][j]
        return Matrix(self.size/2, array)
    def lr(self):
        array = [[0]*(self.size/2) for i in xrange(self.size/2)]
        for i in xrange(self.size/2):
            for j in xrange(self.size/2):
                array[i][j] = self.array[i+self.size/2][j+self.size/2]
        return Matrix(self.size/2, array)

def factors(a):
    return [f for f in xrange(1,a+1) if a%f==0]

if __name__ == "__main__":
    import sys,random
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    oname = sys.argv[2] if len(sys.argv) > 2 else "test"
    mat1 = []
    mat2 = []
    for i in xrange(size):
        mat1.append([])
        mat2.append([])
    for i in xrange(size):
        for j in xrange(size):
            mat1[i].append( random.randint(1,5) )
            mat2[i].append( random.randint(1,5) )
    mat1 = Matrix(size,mat1)
    mat2 = Matrix(size,mat2)
    mat3 = mat1*mat2
    o1 = open(oname+".in", "w")
    o1.write(str(size)+"\n")
    o1.write(str(mat1)+'\n')
    o1.write(str(mat2)+'\n')
    o1.close()
    o2 = open(oname+".correct", "w")
    o2.write(str(mat3)+'\n')
    o2.close()

    # write valid parameters into basename.params
    o3 = open(oname+".params", "w")
    log7=0
    log2=0
    s = size
    while s % 7 == 0:
        s /= 7
        log7 += 1
    while s % 2 == 0:
        s /= 2
        log2 += 1
    maxproc = min(log7*2,log2)
    for log7nproc in xrange(maxproc+1):
        nproc = 7**log7nproc
        for nrec in xrange(log7nproc,log2+1):
            maxbs = size/(7**((log7nproc+1)/2))/(2**nrec)
            for bs in factors(maxbs):
                o3.write("mpirun -n " + str(nproc) + " ./fromfile1 -i tests/" + oname + ".in -b " + str(bs) + " -r " + str(nrec) + " -c tests/" + oname + ".correct\n")
    o3.close()
                         
