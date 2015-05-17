#!/usr/bin/env python

"""
The script for ploting the performance comparison of different 
"""

import os
import tables
import numpy as np
import sys
import pylab as pb

if __name__ == '__main__':
    f_name = sys.argv[1]

    n,ext = os.path.splitext(f_name)

    f = tables.openFile(f_name,'r')
    # method names
    m_names = list(set([a.name.split('_')[0] for a in f.list_nodes('/')]))
    print m_names

    fig = pb.figure()
    ax = fig.add_subplot(111)

    for m_n in m_names:
        ts = f.get_node('/'+m_n+'_time')[:]
        ys = f.get_node('/'+m_n+'_Ybest')[:]     

        ts = ts.mean(axis=0)
        #ts = range(ys.shape[1])
        #ys_mean = ys.mean(axis=0)
        ys_mean = []
        for i in xrange(ys.shape[1]):
             print i
             y = ys_mean[:,i]
             n = (np.isfinite(y)).sum()
             y[np.isnan(y)] = 0
             ys_mean.append(y.sum()/n)
        ys_std = ys.std(axis=0)

        #ax.errorbar(ts, ys_mean, yerr=2*ys_std, label=m_n)
        ax.plot(ts,ys_mean)
        pl.fill_between(ts, ys_mean-2*ys_std, ys_mean+2*ys_std,alpha=0.2,linewidth=4, linestyle='dashdot', antialiased=True)
        
    pb.legend()
    pb.savefig(n+'.pdf')

    f.close()

