import pyscipopt as sc
import sys

if len(sys.argv) != 3:
    print(f"USAGE: {sys.argv[0]} [in filename] [out filename]")
    exit(-1)

filename = sys.argv[1]
outfilename = sys.argv[2]
print(f"Converting {filename} and writing to {outfilename}")

scip = sc.Model()
scip.readProblem(filename=filename, extension="MPS")

conss = list(filter(lambda x: x.isLinear(), scip.getConss(transformed=False)))
vars = [f"{x}" for x in scip.getVars(transformed=False)[:-1]]

nrows = len(conss)
ncols = len(vars)
nnz = sum([len(scip.getValsLinear(cons)) for cons in conss])

outfile = open(outfilename, "w")
outfile.write(f"{nrows} {ncols} {nnz}\n")

for nrow in range(nrows):
    cons = conss[nrow]
    vals = scip.getValsLinear(cons)

    for varname in vals:
        val = vals[varname]
        ncol = vars.index(varname)
        outfile.write(f"{nrow} {ncol} {val}\n")

outfile.close()
