using PyPlot
using OptimTools
using KrylovMethods

include("armijo.jl")




function check_starting_point(f,df,J,x0=[1.0,1.0])

	#BFGS
	tic()
	res1=bfgs(f,df,x0,maxIter=100,lineSearch=my_armijo,atol=0.01,storeInterm=true)
	runtime1=toq()
	his1=res1[3]
	iterations1=size(his1)[1]
	fnc_eval1=sum(his1[:,3])+iterations1
	grad_eval1=iterations1+1
	ode1=fnc_eval1+2*grad_eval1
	@printf "\n\nBFGS \n"
	@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations1 his1[iterations1,1] his1[iterations1,2] 
	@printf "fnc_eval=%d\ngrad_eval=%d\n" fnc_eval1 grad_eval1 
	@printf "ode=%d\nruntime=%f\n" ode1 runtime1
	@printf "The maximum number of line searches was %d\n\n" maximum(his1[:,3])
		

	#Gauss Newton
	tic()
	res2=newtoncg(f,df,J,x0,maxIter=100,lineSearch=my_armijo,atol=0.01,storeInterm=true)
	runtime2=toq()
	#technically possible: ode: #fnc_eval+2*#grad_eval
	his2=res2[3]
	iterations2=size(his2)[1]
	fnc_eval2=sum(his2[:,3])+iterations2
	grad_eval2=iterations2
	hess_eval2=iterations2
	ode2=fnc_eval2+2*grad_eval2+2*hess_eval2
	@printf "\n\nGauss Newton \n"
	@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations2 his2[iterations2,1] his2[iterations2,2] 
	@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval2 grad_eval2 hess_eval2
	@printf "ode=%d\t runtime=%f \n" ode2 runtime2
	@printf "The maximum number of line searches was %d\n\n" maximum(his2[:,3])
	@printf "The maximum number of cg steps was %d\n\n" maximum(his2[:,4])
	
	#Gauss Newton with damped Newton
	tic()
	res3=dampedNewton(f,df,J,x0,maxIter=100,lineSearch=my_armijo,atol=0.01,storeInterm=true)
	runtime3=toq()
	#technically possible: ode: #fnc_eval+2*#grad_eval
	his3=res3[3]
	iterations3=size(his3)[1]
	fnc_eval3=sum(his3[:,3])+iterations3
	grad_eval3=iterations3
	hess_eval3=iterations3
	ode3=fnc_eval3+2*grad_eval3+2*hess_eval3
	@printf "\n\nGauss Newton with damped Newton\n"
	@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations3 his3[iterations3,1] his3[iterations3,2] 
	@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval3 grad_eval3 hess_eval3
	@printf "ode=%d\t runtime=%f \n" ode3 runtime3
	@printf "The maximum number of line searches was %d\n\n" maximum(his3[:,3])
	#@printf "The maximum number of cg steps was %d\n\n" maximum(his3[:,4])

	#return the iterates for the contour plot
	return res1[4],res2[4],res1[3],res2[3]
end

function check_starting_point_noise(nf,ndf,nJ,data2,x0=[1.0,5.0])
	
	
	#BFGS
	tic()
	res1=bfgs(x->nf(x,data2[1]),x->ndf(x,data2[1]),x0,maxIter=10000,lineSearch=my_armijo,atol=0.01,storeInterm=true)
	runtime1=toq()
	his1=res1[3]
	iterations1=size(his1)[1]
	fnc_eval1=sum(his1[:,3])+iterations1
	grad_eval1=iterations1+1
	ode1=fnc_eval1+2*grad_eval1
	@printf "\n\nBFGS \n"
	@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations1 his1[iterations1,1] his1[iterations1,2] 
	@printf "fnc_eval=%d\ngrad_eval=%d\n" fnc_eval1 grad_eval1 
	@printf "ode=%d\nruntime=%f\n" ode1 runtime1
	@printf "The maximum number of line searches was %d\n\n" maximum(his1[:,3])
		

	#Gauss Newton
	tic()
	res2=newtoncg(x->nf(x,data2[1]),x->ndf(x,data2[1]),x->nJ(x,data2[1]),x0,maxIter=10000,lineSearch=my_armijo,atol=0.01,storeInterm=true)
	runtime2=toq()
	#technically possible: ode: #fnc_eval+2*#grad_eval
	his2=res2[3]
	iterations2=size(his2)[1]
	fnc_eval2=sum(his2[:,3])+iterations2
	grad_eval2=iterations2
	hess_eval2=iterations2
	ode2=fnc_eval2+2*grad_eval2+2*hess_eval2
	@printf "\n\nGauss Newton \n"
	@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations2 his2[iterations2,1] his2[iterations2,2] 
	@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval2 grad_eval2 hess_eval2
	@printf "ode=%d\t runtime=%f \n" ode2 runtime2
	@printf "The maximum number of line searches was %d\n\n" maximum(his2[:,3])
	@printf "The maximum number of cg steps was %d\n\n" maximum(his2[:,4])
	
	#return the iterates for the contour plot
	return res1[4],res2[4],res1[3],res2[3]
end

function do_the_noise(nf,ndf,nJ,eps)
		data2=addNoise(data,eps)
		a,b,c,d=check_starting_point_noise(nf,ndf,nJ,data2)
		n=size(a)[2]
		x=a[:,n]
		inp=get_u(x[1],x[2],data[1])
		n=size(b)[2]
		x=b[:,n]
		inp2=get_u(x[1],x[2],data[1])
		plot(grid, data)
		plot(grid,data2)
		plot(grid,map(i->inp[2][i][2],1:100))
		plot(grid,map(i->inp2[2][i][2],1:100))
		title("Displacement for eps=$eps")
		legend(["Original","Noise","BFGS","Gauss-Newton"])
		xlabel("Time t")
		ylabel("Displacement")
end	
	
