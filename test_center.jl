using KrylovMethods
using OptimTools

include("armijo.jl")

include("Dennis-Gay_Welsch.jl")
include("Fletcher_Xu.jl")
include("Brown-Dennis.jl")
include("App_Newton.jl")

include("Freudenstein.jl")
include("classical.jl")
include("Nielsen.jl")
include("Trig.jl")


function test(;x_c=[0.0,10.0,20.0],x_f=[5.0,6.0],x_n=[1.0,1.0,-0.75,0.75],x_t=[25.0,5.0,-1.0,-5.0])
	test_classical(x_0=x_c)
	test_Freudenstein(x_0=x_f)
	test_Nielsen(x_0=x_n)
	test_Trig(x_0=x_t)
end

function test_classical(;x_0=[0.0,10.0,20.0])
	
	
	@printf "The Classical Problem \n\n"
	x=Dennis_Gay_Welsch(Classical,grad_Classical,J_Classical,r_Classical,x_0,3,maxIter=100,atol=10.0^(-8))	
	@printf "Dennis-Gay-Welsch yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]
	
	x=Brown_Dennis(Classical,grad_Classical,J_Classical,r_Classical,x_0,3,10,maxIter=100,atol=10.0^(-8))	
	@printf "Brown-Dennis yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=App_Newton(Classical,grad_Classical,J_Classical,r_Classical,x_0,3,10,maxIter=100,atol=10.0^(-8))	
	@printf "App-Newton yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=Fletcher_Xu(Classical,grad_Classical,J_Classical,r_Classical,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Fletcher-Xu yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "where we took %d Gauss and %d bfgs steps\n" x[4][1] x[4][2] 
	@printf "Function eval: %d \t J eval: %d\n" x[5][1] x[5][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[5][3] x[5][4] x[5][5]

	x=my_Gauss_Newton(Classical,grad_Classical,J_Classical,r_Classical,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Gauss-Newton yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=my_bfgs(Classical,grad_Classical,J_Classical,r_Classical,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "MBFGS yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=bfgs(Classical,grad_Classical,x_0,maxIter=100,atol=10.0^(-8))
	iter=size(x[3])[1]	
	fnc_eval=sum(x[3][:,3])+iter
	J_eval=iter+1
	mat_vec_prod=J_eval+3*iter
	mat_mat_prod=4*iter
	@printf "BFGS yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" size(x[3])[1] x[3][size(x[3])[1],2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval J_eval
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" mat_mat_prod mat_vec_prod 0	
	
	
	x=modnewton(Classical,grad_Classical,Hess_Classical,x_0,maxIter=100,atol=10.0^(-8))
	his=x[3]
	iter=size(x[3])[1]
	factor=sum(his[:,4])
	fnc_eval=iter+sum(his[:,3])
	@printf "ModNewton yields a final approximation of [%f,%f,%f]\n" x[1][1] x[1][2] x[1][3]
	@printf "in %d iterations with a gradient norm of %0.2e\n" iter x[3][iter,2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval iter
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" 0 iter factor	
end

function test_Freudenstein(;x_0=[6.0,6.0])
	# Freudenstein

	@printf "Freudenstein Function\n\n"
	x=Dennis_Gay_Welsch(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,2,maxIter=100,atol=10.0^(-8))	
	@printf "Dennis-Gay-Welsch yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]
	
	x=Brown_Dennis(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,2,2,maxIter=100,atol=10.0^(-8))	
	@printf "Brown-Dennis yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=App_Newton(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,2,2,maxIter=100,atol=10.0^(-8))	
	@printf "App_Newton yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=Fletcher_Xu(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Fletcher-Xu yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "where we took %d Gauss and %d bfgs steps\n\n" x[4][1] x[4][2]
	@printf "Function eval: %d \t J eval: %d\n" x[5][1] x[5][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[5][3] x[5][4] x[5][5]

	x=my_Gauss_Newton(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Gauss-Newton yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=my_bfgs(Freudenstein,grad_Freudenstein,J_Freudenstein,r_Freudenstein,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "M-BFGS yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=bfgs(Freudenstein,grad_Freudenstein,x_0,maxIter=100,atol=10.0^(-8))	
	iter=size(x[3])[1]	
	fnc_eval=sum(x[3][:,3])+iter
	J_eval=iter+1
	mat_vec_prod=J_eval+3*iter
	mat_mat_prod=4*iter
	@printf "BFGS yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n" size(x[3])[1] x[3][size(x[3])[1],2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval J_eval
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" mat_mat_prod mat_vec_prod 0	
	
	x=modnewton(Freudenstein,grad_Freudenstein,Hess_Freudenstein,x_0,maxIter=100,atol=10.0^(-8))	
	his=x[3]
	iter=size(x[3])[1]
	factor=sum(his[:,4])
	fnc_eval=iter+sum(his[:,3])
	@printf "ModNewton yields a final approximation of [%f,%f]\n" x[1][1] x[1][2]
	@printf "in %d iterations with a gradient norm of %0.2e\n" iter x[3][iter,2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval iter
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" 0 iter factor	
end

function test_Nielsen(;x_0=[1.0,1.0,-0.75,0.75],x_1=[1.0,2.0,-2.0,1.0])#x_0=[1.0,1.0,-0.75,0.75])
	
	
	@printf "The Nielsen Problem part 1\n\n"
	x=Dennis_Gay_Welsch(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,4,maxIter=100,atol=10.0^(-8))	
	@printf "Dennis-Gay-Welsch yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]
	
	x=Brown_Dennis(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,4,10,maxIter=100,atol=10.0^(-8))	
	@printf "Brown-Dennis yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=App_Newton(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,4,10,maxIter=100,atol=10.0^(-8))	
	@printf "App_Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=Fletcher_Xu(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Fletcher-Xu yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "where we took %d Gauss and %d bfgs steps\n\n" x[4][1] x[4][2]
	@printf "Function eval: %d \t J eval: %d\n" x[5][1] x[5][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[5][3] x[5][4] x[5][5]

	x=my_Gauss_Newton(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Gauss-Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=my_bfgs(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "M-BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=bfgs(Nielsen,grad_Nielsen,x_0,maxIter=100,atol=10.0^(-8))	
	iter=size(x[3])[1]	
	fnc_eval=sum(x[3][:,3])+iter
	J_eval=iter+1
	mat_vec_prod=J_eval+3*iter
	mat_mat_prod=4*iter
	@printf "BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" size(x[3])[1] x[3][size(x[3])[1],2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval J_eval
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" mat_mat_prod mat_vec_prod 0	

	x=modnewton(Nielsen,grad_Nielsen,Hess_Nielsen,x_0,maxIter=100,atol=10.0^(-8))	
	his=x[3]
	iter=size(x[3])[1]
	factor=sum(his[:,4])
	fnc_eval=iter+sum(his[:,3])
	@printf "ModNewton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" iter x[3][iter,2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval iter
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" 0 iter factor	


	@printf "The Nielsen Problem part 2\n\n"
	x=Dennis_Gay_Welsch(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,4,maxIter=100,atol=10.0^(-8))	
	@printf "Dennis-Gay-Welsch yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]
	
	x=Brown_Dennis(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,4,10,maxIter=100,atol=10.0^(-8))	
	@printf "Brown-Dennis yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=App_Newton(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,4,10,maxIter=100,atol=10.0^(-8))	
	@printf "App_Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=Fletcher_Xu(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,maxIter=100,atol=10.0^(-8),tau=0.05)	
	@printf "Fletcher-Xu yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "where we took %d Gauss and %d bfgs steps\n\n" x[4][1] x[4][2]
	@printf "Function eval: %d \t J eval: %d\n" x[5][1] x[5][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[5][3] x[5][4] x[5][5]

	x=my_Gauss_Newton(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,maxIter=100,atol=10.0^(-8))	
	@printf "Gauss-Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=my_bfgs(Nielsen,grad_Nielsen,J_Nielsen,r_Nielsen,x_1,maxIter=100,atol=10.0^(-8))	
	@printf "M-BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=bfgs(Nielsen,grad_Nielsen,x_1,maxIter=100,atol=10.0^(-8))
	iter=size(x[3])[1]	
	fnc_eval=sum(x[3][:,3])+iter
	J_eval=iter+1
	mat_vec_prod=J_eval+3*iter
	mat_mat_prod=4*iter
	@printf "BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" size(x[3])[1] x[3][size(x[3])[1],2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval J_eval
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" mat_mat_prod mat_vec_prod 0	

	x=modnewton(Nielsen,grad_Nielsen,Hess_Nielsen,x_1,maxIter=100,atol=10.0^(-8))	
	his=x[3]
	iter=size(x[3])[1]
	factor=sum(his[:,4])
	fnc_eval=iter+sum(his[:,3])
	@printf "ModNewton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" iter x[3][iter,2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval iter
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" 0 iter factor	
end
	
function test_Trig(;x_0=[25.0,5.0,-1.0,-5.0])
	
	
	@printf "The Trig Problem \n\n"
	x=Dennis_Gay_Welsch(Trig,grad_Trig,J_Trig,r_Trig,x_0,4,maxIter=100,atol=10.0^(-8))	
	@printf "Dennis-Gay-Welsch yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]
	
	x=Brown_Dennis(Trig,grad_Trig,J_Trig,r_Trig,x_0,4,20,maxIter=100,atol=10.0^(-8))	
	@printf "Brown-Dennis yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=App_Newton(Trig,grad_Trig,J_Trig,r_Trig,x_0,4,20,maxIter=100,atol=10.0^(-8))	
	@printf "App_Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=Fletcher_Xu(Trig,grad_Trig,J_Trig,r_Trig,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Fletcher-Xu yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" x[2] x[3]
	@printf "where we took %d Gauss and %d bfgs steps\n\n" x[4][1] x[4][2]
	@printf "Function eval: %d \t J eval: %d\n" x[5][1] x[5][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[5][3] x[5][4] x[5][5]

	x=my_Gauss_Newton(Trig,grad_Trig,J_Trig,r_Trig,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "Gauss-Newton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=my_bfgs(Trig,grad_Trig,J_Trig,r_Trig,x_0,maxIter=100,atol=10.0^(-8))	
	@printf "M-BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n\n" x[2] x[3]
	@printf "Function eval: %d \t J eval: %d\n" x[4][1] x[4][2]
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" x[4][3] x[4][4] x[4][5]

	x=bfgs(Trig,grad_Trig,x_0,maxIter=100,atol=10.0^(-8))	
	iter=size(x[3])[1]	
	fnc_eval=sum(x[3][:,3])+iter
	J_eval=iter+1
	mat_vec_prod=J_eval+3*iter
	mat_mat_prod=4*iter
	@printf "BFGS yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" size(x[3])[1] x[3][size(x[3])[1],2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval J_eval
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" mat_mat_prod mat_vec_prod 0	
	
	x=modnewton(Trig,grad_Trig,Hess_Trig,x_0,maxIter=100,atol=10.0^(-8))	
	his=x[3]
	iter=size(x[3])[1]
	factor=sum(his[:,4])
	fnc_eval=iter+sum(his[:,3])
	@printf "ModNewton yields a final approximation of [%f,%f,%f,%f]\n" x[1][1] x[1][2] x[1][3] x[1][4]
	@printf "in %d iterations with a gradient norm of %0.2e\n" iter x[3][iter,2]
	@printf "Function eval: %d \t J eval: %d\n" fnc_eval iter
	@printf "MxM: %d \t Mxv: %d \t M_fac: %d \n\n" 0 iter factor	
end

