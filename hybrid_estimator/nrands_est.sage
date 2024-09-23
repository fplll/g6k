from batchCVP import batchCVPP_cost
from sage.plot.scatter_plot import ScatterPlot

dim = 56

data = []
for M in range(200, 1151, 50):
    log_M = log(M, 2).n()
    prob, T = batchCVPP_cost(dim, log_M, sqrt(4/3.), 1)
    data.append([M,min(1,(prob**dim)*M)])

print(data)


g = Graphics()
g+= list_plot(data, marker='+', color='darkgreen')
g.show()
g.save("nrands_"+str(dim)+".png")
