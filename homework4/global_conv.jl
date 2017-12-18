using PyPlot
using OptimTools
using KrylovMethods

include("mesh.jl")
include("armijo.jl")


function check_convergence(f,df,J)
	#Create the required mesh for the exercise
	mesh=create_mesh(20,[0.1 5],[0.1 5])


	#create variables to store results
	region=zeros(20,20)
	iterations=zeros(20,20)
	runtime=zeros(20,20)

	#Apply NewtonCG to each point
	for i = 1:20
  		for j = 1:20
			@printf "i=%d\n" i
			tic()
    			res=newtoncg(f,df,J,vec(mesh[i,j,:]),maxIter=100,lineSearch=my_armijo,atol=10.0^(-2))
			runtime[i,j]=toq()
			his=res[3]
    			iterations[i,j]=size(his)[1]
			if abs(his[iterations[i,j],2])<10.0^(-2) 
				region[i,j]=1
			end 
			if  f(res[1])[1]<0
				region[i,j]=-1
			end
			@printf "j=%d\n" j
			@printf "Converges=%d\n" region[i,j]
  		end
	end
	
	return runtime,iterations,region
end



