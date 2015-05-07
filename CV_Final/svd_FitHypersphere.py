"""
        FitHypersphere.py
        
        fit_hypersphere(collection of tuples or lists of real numbers)
        will return a hypersphere of the same dimension as the tuples:
                (radius, (center))

        using the Hyper (hyperaccurate) algorithm of 
        Ali Al-Sharadqah and Nikolai Chernov
        Error analysis for circle fitting algorithms
        Electronic Journal of Statistics
        Vol. 3 (2009) 886-911
        DOI: 10.1214/09-EJS419

        generalized to n dimensions

        Mon Apr 23 04:08:05 PDT 2012 Kevin Karplus

        Note: this version using SVD works with Hyper, Pratt, and Taubin methods.
        If you are not familiar with them, Hyper is probably your best choice.
        
        
        Creative Commons Attribution-ShareAlike 3.0 Unported License.
        http://creativecommons.org/licenses/by-sa/3.0/
"""             

import numpy as np
from numpy import linalg
from sys import stderr
from math import sqrt




def fit_hypersphere(data, method="Hyper"):
    """returns a hypersphere of the same dimension as the 
        collection of input tuples
                (radius, (center))
    
       Methods available for fitting are "algebraic" fitting methods
        Hyper   Al-Sharadqah and Chernov's Hyperfit algorithm
        Pratt   Vaughn Pratt's algorithm
        Taubin  G. Taubin's algorithm
    
       The following methods, though very similar, are not implemented yet,
          because the contraint matrix N would be singular, 
          and so the N_inv computation is not doable.
       
        Kasa    Kasa's algorithm
    """
    num_points = len(data)
#    print >>stderr, "DEBUG: num_points=", num_points
    
    if num_points==0:
        return (0,None)
    if num_points==1:
        return (0,data[0])
    dimen = len(data[0])        # dimensionality of hypersphere
#    print >>stderr, "DEBUG: dimen=", dimen
    
    if num_points<dimen+1:
        raise ValueError(\
            "Error: fit_hypersphere needs at least {} points to fit {}-dimensional sphere, but only given {}".format(dimen+1,dimen,num_points))
    
    # central dimen columns of matrix  (data - centroid)
    central = np.matrix(data, dtype=float)      # copy the data
    centroid = np.mean(central, axis=0)
    for row in central:
        row -= centroid
#    print >>stderr, "DEBUG: central=", repr(central)

    # squared magnitude for each centered point, as a column vector
    square_mag= [sum(a*a for a in row.flat) for row in central] 
    square_mag = np.matrix(square_mag).transpose()
#    print >>stderr, "DEBUG: square_mag=", square_mag
    
    if method=="Taubin":
        # matrix of normalized squared magnitudes, data
        mean_square = square_mag.mean()
        data_Z = np.bmat( [[(square_mag-mean_square)/(2*sqrt(mean_square)), central]])
    #    print >> stderr, "DEBUG: data_Z=",data_Z
        u,s,v = linalg.svd(data_Z, full_matrices=False)
        param_vect= v[-1,:]
        params = [ x for x in np.asarray(param_vect)[0]]        # convert from (dimen+1) x 1 matrix to list
        params[0] /= 2*sqrt(mean_square)
        params.append(-mean_square*params[0])
        params=np.array(params)
        
    else:
        # matrix of squared magnitudes, data, 1s
        data_Z = np.bmat( [[square_mag, central, np.ones((num_points,1))]])
    #    print >> stderr, "DEBUG: data_Z=",data_Z

        # SVD of data_Z
        # Note: numpy's linalg.svd returns data_Z = u * s * v
        #         not u*s*v.H as the Release 1.4.1 documentation claims.
        #         Newer documentation is correct.
        u,s,v = linalg.svd(data_Z, full_matrices=False)
    #    print >>stderr, "DEBUG: u=",repr(u)
    #    print >>stderr, "DEBUG: s=",repr(s)
    #    print >>stderr, "DEBUG: v=",repr(v)
    #    print >>stderr, "DEBUG: v.I=",repr(v.I)

        if s[-1]/s[0] < 1e-12:
            # singular case
            # param_vect as (dimen+2) x 1 matrix
            param_vect = v[-1,:]
            # Note: I get last ROW of v, while Chernov claims last COLUMN,
            # because of difference in definition of SVD for MATLAB and numpy

    #        print >> stderr, "DEBUG: singular, param_vect=", repr(param_vect)
    #        print >> stderr, "DEBUG: data_Z*V=", repr(data_Z*v)
    #        print >> stderr, "DEBUG: data_Z*VI=", repr(data_Z*v.I)
    #        print >> stderr, "DEBUG: data_Z*A=", repr(data_Z*v[:,-1])
        else:    
            Y = v.H*np.diag(s)*v
            Y_inv = v.H*np.diag([1./x for x in s])*v
    #        print >>stderr, "DEBUG: Y=",repr(Y)
    #        print >>stderr, "DEBUG: Y.I=",repr(Y.I), "\nY_inv=",repr(Y_inv)
            #Ninv is the inverse of the constraint matrix, after centroid has been removed
            Ninv = np.asmatrix(np.identity(dimen+2, dtype=float))
            if method=="Hyper":
                Ninv[0,0] = 0
                Ninv[0,-1]=0.5
                Ninv[-1,0]=0.5
                Ninv[-1,-1] = -2*square_mag.mean()
            elif method=="Pratt":
                Ninv[0,0] = 0
                Ninv[0,-1]=-0.5
                Ninv[-1,0]=-0.5
                Ninv[-1,-1]=0
            else: 
                raise ValueError("Error: unknown method: {} should be 'Hyper', 'Pratt', or 'Taubin'")
    #        print >> stderr, "DEBUG: Ninv=", repr(Ninv)

            # get the eigenvector for the smallest positive eigenvalue
            matrix_for_eigen = Y*Ninv*Y
    #   print >> stderr, "DEBUG: {} matrix_for_eigen=\n{}".format(method, repr(matrix_for_eigen))
            eigen_vals,eigen_vects = linalg.eigh(matrix_for_eigen)
    #   print >> stderr, "DEBUG: eigen_vals=", repr(eigen_vals)
    #   print >> stderr, "DEBUG: eigen_vects=", repr(eigen_vects)

            positives = [x for x in eigen_vals if x>0]
            if len(positives)+1 != len(eigen_vals):
                # raise ValueError("Error: for method {} exactly one eigenvalue should be negative: {}".format(method,eigen_vals))
                print>>stderr, "Warning: for method {} exactly one eigenvalue should be negative: {}".format(method,eigen_vals)
            smallest_positive = min(positives)
    #    print >> stderr, "DEBUG: smallest_positive=", smallest_positive
            # chosen eigenvector as 1 x (dimen+2) matrix
            A_colvect =eigen_vects[:,list(eigen_vals).index(smallest_positive)]
    #        print >> stderr, "DEBUG: A_colvect=", repr(A_colvect)
            # now have to multiply by Y inverse
            param_vect = (Y_inv*A_colvect).transpose()
    #        print >> stderr, "DEBUG: nonsingular, param_vect=", repr(param_vect)        
        params = np.asarray(param_vect)[0]  # convert from (dimen+2) x 1 matrix to array of (dimen+2)

        
#    print >> stderr, "DEBUG: params=", repr(params)
    radius = 0.5* sqrt( sum(a*a for a in params[1:-1])- 4*params[0]*params[-1])/abs(params[0])
    center = -0.5*params[1:-1]/params[0]
#y    print >> stderr, "DEBUG: center=", repr(center), "centroid=", repr(centroid)
    center += np.asarray(centroid)[0]
    return (radius,center)
    
