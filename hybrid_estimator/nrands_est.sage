from batchCVP import batchCVPP_cost
from sage.plot.scatter_plot import ScatterPlot

dim = 60

cdb_size = (8730)**(1./dim)
print("cdb_size:", cdb_size, sqrt(4/3).n())


data = []
for M in range(10, 140, 10):
    log_M = log(M, 2).n()
    prob, T = batchCVPP_cost(dim, log_M, cdb_size, 1.025) #change sqrt(4/3) to the value cdb_size()^{1/dim}
    data.append([M,min(1,(prob**dim)*M)])

print(data)


g = Graphics()
g+= list_plot(data, marker='+', color='darkgreen')
g.show()
g.save("nrands_"+str(dim)+".png")
