class mockSimulation:

        ## public
        def __init__(self,D,d,fun_list,grad_fun_list):
                self.theta_mean = [0, 0]
                self.theta_std = [0, 0]
                
                self.fun_list = fun_list
                self.grad_fun_list = grad_fun_list
                self.N = 1# number of parallel simulations
                self.D = D
                self.d = d
                self.n_OP = D + len(fun_list)
                
                
                #tuning parameter for updating weights
                self.delta = .25
                #number of clusters in C_p
                self.num_clusters_lc =2
                #Potential Parameters:
                self.a=10
                
                self.b=250
                
                self.rad=2 #rad is radius of circle
                
                #potential looks like
                #d is distance to circle
                #V = -b*t.exp(-a*d)
               
       
        
        
        #it should input a list of 10=D lists, which contain the initial points.
        #should return a tuple of 10 lists, each list is a list of numpy arrays that are 10d, and each array only has 1 row
        
        #this was my first time using pytorch so I'm sure there's a better way to do automatic differentiation using it
        #you can change V and d(if you want to use a different potential)
        def run_noPlt(self, inits, nstepmax = 10):
                import numpy as np
                import torch as t
                import time 
                from scipy.interpolate import interp1d
                #import matplotlib.pyplot as plt
                           
                # temperature?
                D = self.D
                mu = 3
                #potential parameters
                a = self.a
                b = self.b 
                rad = self.rad
                # time-step 
                h = 1e-4
                
                x = []
                # initialization
                for i in range(D):
                    x.append(t.tensor(inits[i], requires_grad=True))
                 
                z = np.ones(len(inits[0]))
                
                external_grad = t.tensor(np.ndarray.tolist(z))
               
                trj_all = []
                
                
                for nstep in range(int(nstepmax)):
                        
                        
                        #distance to circle squared
                        d = t.pow(t.sqrt(t.pow(x[0],2)+t.pow(x[1],2)) - rad,2)
                        for i in range(2,D):
                            d = d + t.pow(x[i],2)
                       
                       

                        V = -b*t.exp(-a*d)
                      
                        V.backward(external_grad)
                        
                        dVx = []
                        for i in range(D):
                            dVx.append(x[i].grad)
                        
                        
                        
                        #the line below is making x and y into 2d tensors rather than 1d(because of t.randn(1,len(inits_x))
                       
                        
                        for i in range(D):
                            x[i] = x[i] - h*dVx[i] + (h*mu)**.5*t.randn(1,len(inits[0]))
                            x[i] = x[i].detach().numpy().astype('float')
                            trj_all.append([])
                            trj_all[i].append(x[i])
                           
                        
                        #you need to make x a list so you can make a tensor out of it
                        #x[0] because x is a 2d array with one row so you have to access that one row before making it a list
                        for i in range(D):
                            x[i] = np.ndarray.tolist(x[i][0])
                            x[i] = t.tensor(x[i], requires_grad=True)
                        
                        
                        #reset V and d, I'm not sure if necessary
                        V = None
                        d = None
                        
                        #define a tuple from a list
                        toReturn = []
                        for i in range(D):
                            toReturn.append(trj_all[i])
                            
                        toReturn1 = tuple(toReturn)
                return toReturn1  
            
            
        def PreAll(self, trj):
                """
                This just changes the shape/data structure type of the trajectory
                """
                import numpy as np
                comb_trj = []
                for theta in range(len(trj)):
                        comb_trj.append(np.concatenate(np.concatenate(trj[theta])))
                trj_Sp = np.array(comb_trj) # pick all
                
                return trj_Sp


        
        #N is the number of clusters we want to select for the least count part
        #starting_n is the numbers of points we want to return(or at least that many)
        
        #In summary, PreSamp is first clustering trj as observations in R^D, then 
        #picking the num_clusters_cl rarest ones, then collecting all of those observations. 
        #From those observations it will make sure
        #to duplicate the observations so we have at least starting_n observations in the trajectories that we return.
        
        #if n_cl is specified when you call, then that overrides the computation of n_cl. 
        #If it's not specified, then it is computed
        #starting_n is the number of points we require that we have in the least clusters(if too low, it returns
        #duplicates of the points we do have)
        def PreSamp(self, trjs, starting_n=10, n_cl = None):
                """
                Pre-Sampling:
                        choose states with minimum counts or newly discovered states
                        
                output:
                        trj with shape of [[X1s][X2s]...[XDs]
                """
                import numpy as np
                
                N = self.num_clusters_lc
                
                
                d = self.d
                
                
                #each observation is a row after transposing
                
                trjs = trjs.T
                
                
                #parameters
                #for computing n_cl
                c = .07
                gamma = .7
                
                #for computing n_cl_prime
                C = 2
                
                #num observations
                n = trjs.shape[0]
                #print(n)
                if n_cl == None:
                    #calculate n_cl
                    n_cl = np.floor(c*n**(gamma*d)).astype(int)
                
                n_cl_prime= (np.floor(n_cl*np.log(n_cl)) + C).astype(int)
                #print(n_cl_prime)
                #print(n_cl)
                #sample n_cl^prime points uniformly from all data
                sample = []
                for i in range(n_cl_prime):
                    sample.append( np.random.randint(0,n))
                    
                selected_points = trjs[sample,:]
                
                #do FFT on the n_cl_prime selected points to choose n_cl of them
                clusters_0 = self.FFT(selected_points,n_cl)
                
                from sklearn.cluster import KMeans
                cluster_obj = KMeans(n_clusters=n_cl,init = clusters_0, max_iter = 1,n_init=1)
                cluster_obj.fit(trjs)
                cl_trjs = cluster_obj.labels_
                cl_centroids = cluster_obj.cluster_centers_
                #print(cl_centroids)
                #print(cl_trjs)
                #cl_trjs is an array of numbers of length number of data points(which are in R^D), where each 
                #number is between 0 and myn_clusters-1, which tells you which cluster that observation fell into
                #after clustering
                
                
                
                unique, counts = np.unique(cl_trjs, return_counts=True)
                #unique is just the unique numbers it finds in the given array, so the numbers 0 to myn_clusters-1
                #I think always. counts is an array of the numbers of observations in each cluster
                
                #leastPop are the N rarest clusters, I'm not sure how it decides ties(or if that even matters)
                #leastPop is a numpy array
                leastPop = counts.argsort()[:N]
                #print(leastPop)
                #init_cl is a list(same entries as leastPop)
                init_cl = [unique[i] for i in leastPop]
                #print(init_cl)
                
                #init_index has the indices of the observations(in R^D) which belong to the N rarest clusters
                #init_trj_xy is putting those observations into an list as rows
                #trjs_Sp is the numpy array of the transpose of that init_trj_xy so the observations are columns,
                
                #init_index = []
                init_trj_xy = []
                for i in range(len(cl_trjs)):
                        if cl_trjs[i] in init_cl:
                                #print(cl_trjs[i])
                                #init_index.append(i)
                                init_trj_xy.append(trjs[i,:])
                trj_Sp = np.array(init_trj_xy)
                
                #trj_Sp has each observation as a column
                trj_Sp = trj_Sp.T
                
                #this just duplicates trj_Sp and appends it to the end(so the number of rows doubles every time)
                #until you have more observations than starting_n(just in case there weren't enough points in the least count clusters
                while len(trj_Sp[0])<starting_n:
                        print('trj_Sp<starting_n')
                        print(len(trj_Sp[0]), starting_n)
                        trj_Sp = np.hstack((trj_Sp,trj_Sp))
                 
                return (trj_Sp,cl_trjs, cl_centroids,counts)
            
        #n_cl_prime observations, returns n_cl points found using FFT
        #selected_points has observations as rows
        def FFT(self,selected_points,n_cl):
            #note D is not the same as the dimension like self.D, here it's a distance matrix
            #this is n_cl_prime
            n_cl_prime = selected_points.shape[0]
            #find first center
            import numpy as np
            mu_1 = np.random.randint(0,n_cl_prime)
            
            mus = []
            mus.append(mu_1)
            aval_mus = []
            for i in range(n_cl_prime):
                aval_mus.append(i)
            aval_mus.remove(mu_1)
            
            #diagonal is infinite
            D = np.ones((n_cl_prime,n_cl_prime))*np.inf
            for i in range(n_cl_prime):
                for j in range(i+1,n_cl_prime):
                    D[i,j] = np.linalg.norm(selected_points[i,:] - selected_points[j,:])
                    D[j,i] = D[i,j]
            
            for k in range(1,n_cl):
                #an index corresponding to max min distance point so far
                max_sofar = aval_mus[0]
                min_dist = np.amin(D[max_sofar,np.array(mus)])
                
                for i in aval_mus[1:len(aval_mus)]:
                    if np.amin(D[i,np.array(mus)]) > min_dist:
                        max_sofar = i
                        min_dist = np.amin(D[max_sofar,np.array(mus)])
                aval_mus.remove(max_sofar)
                mus.append(max_sofar)
            #print(mus)
            
            centers = selected_points[np.array(mus),:]
            
            #print(centers)
            return centers
        
        def map(self, trjs):
                # map coordinate space to reaction coordinates space
                #because we use D coordinate projections, we just have to add the new CVs(in this case, just the angle)
                import numpy as np
                
                fun_list = self.fun_list
                trj_theta = trjs
                n_OP = self.n_OP
                D = self.D
                #I want to add a new row
                new_row = np.zeros((1,trjs.shape[1]))
                #print(new_row)
                for k in range(n_OP-D):
                    for i in range(trjs.shape[1]):
                        new_row[0,i] = fun_list[k](trjs[:,i])
                    trj_theta = np.vstack((trj_theta,new_row))
                
                return trj_theta

        def reward_state(self, S, theta_mean, theta_std, W_):
                r_s = 0
                for k in range(len(W_)):
                    r_s = r_s + W_[k]*(abs(S[k] - theta_mean[k])/theta_std[k])#No direction
                   
                return r_s  

        def reward_state_withoutStd(self, S, theta_mean, theta_std, W_):
                r_s = 0
                for k in range(len(W_)):
                    r_s = r_s + W_[k]*abs(S[k] - theta_mean[k])#No direction
                                       
                return r_s

        #the mean and std are basically global, and they're just lists of the means and stds of the K OPs each
        def updateStat(self, trj_Sp_theta):      
                import numpy as np
                
                theta_mean = []
                theta_std = []
                for theta in range(len(trj_Sp_theta)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                self.theta_std = theta_std
                self.theta_mean = theta_mean
        
        
        
        def updateW(self, trjs,cl_trjs,cl_centroids,cl_counts, W_0):
            import numpy as np
            
            #points_in_clusters = []
            D = self.D
            d = self.d
            n_OP = self.n_OP
            Vs = []
            #these will be the gradients of all the OPs, a so matrix for each centroid
            Xs= []
            ones = np.ones(D)
            init_gradients = np.diag(ones)
            #thetagrad = np.zeros(D,1)
            for i in range(len(cl_centroids)):
                points = trjs[:,cl_trjs==i]
                #print(points)
                
                #first we need to subtract off the means
                B = points.T
                #now observations are rows
                theta_means = []
                for j in range(B.shape[1]):
                    theta_means.append(np.mean(B[:,j]))
                    B[:,j] = B[:,j] - theta_means[j]*np.ones(B.shape[0])
                #points_in_clusters.append(points)
                u, s, vt = np.linalg.svd(B, full_matrices=True) #data = U Sigma V^T
                Vs.append(vt.T[:,0:d])
                #print(Vs[i])
                gradients = init_gradients
                #if there are more OPs than just the coordinate projections, then here you'll add those to gradients as columns
                #(they depend on i)
                #this is computing the gradient of OP_j on centroid i
                for j in range(n_OP-D):
                    new_gradient = np.zeros((D,1))
                    for k in range(D):
                        new_gradient[k] = self.grad_fun_list[j][k](cl_centroids[i])
                    gradients = np.hstack((gradients,new_gradient))
                
                #print(gradients)
                Xs.append(gradients)
            #print(Vs)
            #now we normalize the gradients(only need to normalize the ones that aren't coordinate projections):
            for j in range(n_OP-D):
                #print(j)
                normalized_grad = np.zeros((D,1))
                constant = 0
                #compute square of normalizing constant
                for i in range(len(cl_centroids)):
                    constant =constant + np.linalg.norm(Xs[i][:,j+D])
                
                #with grads with normalized ones
                for i in range(len(cl_centroids)):
                    #print(Xs[i][:,j+D])
                    Xs[i][:,j+D] = Xs[i][:,j+D]/(constant/len(cl_centroids))
                    
            
            A = np.zeros((len(W_0) , len(W_0) ))
            
            num_obs = trjs.shape[1]
            for i in range(len(cl_centroids)):
                
                A = A + Xs[i].T@np.outer(Vs[i],Vs[i])@Xs[i]*(cl_counts[i]/num_obs)
                
            u, s, vt = np.linalg.svd(A)
            
            what = vt.T[:,0]
            
            W_1 = []
            for i in range(len(W_0)):
                W_1.append(what[i]**2)
            #print(W_1)
            return W_1
           
            
        def findStarting(self, trj_Sp_theta, trj_Sp, W_1, starting_n=10):
                # get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
                import numpy as np
                
                theta_mean = []
                theta_std = []
                for theta in range(len(W_1)):
                        theta_mean.append(np.mean(trj_Sp_theta[theta]))
                        theta_std.append(np.std(trj_Sp_theta[theta]))
                        
                ranks = {}
                for state_index in range(len(trj_Sp_theta[0])):
                        #print(len(trj_Sp_theta[0]))
                        state_theta = trj_Sp_theta[:,state_index]
                        #this is step 9 from the paper, which is basically doing equation (1) from the paper but instead of
                        #evaluating at a specific cluster, it evaluates at a given state, and the mean/st.dv. isn't calculated
                        #over all the data so far/all the clusters, it's just calculated over the rarest clusters.
                        #So it's rewarding states which caused the important OPs to deviate the most from their average
                        #behavior on the rarest clusters
                        #i.e. reward high variation and which OPs are important(based on the weights)
                        r = self.reward_state_withoutStd( state_theta, theta_mean, theta_std, W_1)
                        
                        ranks[state_index] = r
                #Note that we're only looking at the first starting_n points here, while we calculated the means over all
                #the points.
                #We're picking the first starting_n points with the highest reward functions
                newPoints_index0 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[0:starting_n] 
                newPoints_index = np.array(newPoints_index0)[:,0]   
                
                n_coord = len(trj_Sp)
                                     
                newPoints = []
                #notice we're getting exactly starting_n points as we should
                for coord in range(n_coord):
                          newPoints.append([trj_Sp[coord][int(i)] for i in newPoints_index])                                
          
                return newPoints
        
        
                
        
        #thetas vary from -pi to pi. For that reason, it's not obvious how to compute what percentage of the circle has been
        #explored so far(you can't just do max angle - min angle). I partition [-pi,pi] into 100 parts and then see how many
        #are filled to do this.
        def percentage_explored(self,thetas):
            import numpy as np
            num_bins = 100
            left_boundaries =np.linspace(-np.pi,np.pi, num=num_bins)
            #print(left_boundaries)
            #print(left_boundaries.shape)
            undiscovered = []
            for i in range(num_bins-1):
                undiscovered.append(i)
            for i in range(len(thetas)):
                for j in undiscovered:
                    if thetas[i]>=left_boundaries[j] and thetas[i]<left_boundaries[j+1]:
                        undiscovered.remove(j)
                        break
            percentage = 1-len(undiscovered)/num_bins
            print(percentage)
            return percentage
        
        def pltFinalPoints(self, trjs_theta):
                import numpy as np
                import matplotlib.pyplot as plt
                x = np.array(trjs_theta[0])
                y = np.array(trjs_theta[1])
                
                a = self.a
                b= self.b 
                rad = self.rad
                
                def V0(x,y):
                    
                    #distance to circle
                    d = np.square(np.sqrt(np.square(x)+np.square(y)) - rad)
                    V = -b*np.exp(-d*a)
                    return V
                
                zxx = np.mgrid[-(rad+1):(rad+1):0.01]
                zyy = np.mgrid[-(rad+1):(rad+1):0.01]
                xx, yy = np.meshgrid(zxx, zyy)
                V1 = V0(xx,yy)

                

                fig = plt.figure()
                ax = fig.add_subplot(111)
                print(V1.shape)

                ax.contourf(xx,yy,V1, 40)

                plt.xlabel('x')
                plt.ylabel('y')
                plt.plot(x, y, 'o', color='white', alpha=0.2, mec="black")

                #plt.scatter(x, y, s=.1)
                #plt.savefig('500TSbetterclusteringD10',dpi =500)
                plt.show()


