using KrylovMethods
using OptimTools
include("check_derivative.jl")
include("approxHessian.jl")

#for plots

font1 = ["family"=>"serif",
    "color"=>"black",
    "weight"=>"normal",
    "size"=>17]



#solver for optimal control problem

N=400
T=1
y0=0
h=T/(N-1)

function get_y(u)
  y=zeros(N)
  y[1]=y0
  for j = 2:N
    y[j]=y[j-1]+h*u[j-1]*y[j-1]+h^3*(j-1)^2
  end
  return y
end

function f(u,y)
  ret=0
  for j = 1:N
    ret+=(y[j]-3.0)^2+1/2*u[j]^2
  end
  return h*ret
end

function get_p(u,y)
  p=zeros(N)
  for j = 1:(N-1)
    p[N-j]=p[N-j+1]+h*p[N-j+1]*u[N-j+1]+2*h*(y[N-j+1]-3)
  end
  return p
end

function get_gradf(u,y,p)
  res=zeros(N)
  for j = 1:N
    res[j]=p[j]*y[j]+u[j]
  end
  return h*res
end

function fnc(u)
  return f(u,get_y(u))
end

function deriv(u)
  y=get_y(u)
  return get_gradf(u,y,get_p(u,y))
end


#Check Derivative and Hessian
#check_derivative(fnc,deriv,N,1)
#check_hessian(fnc,deriv,u->approx_hessian(deriv,u),N)

#first starting point
u_0=10*ones(N)

@time res=bfgs(fnc,deriv,u_0,maxIter=100)
u_1=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+1
hess_eval=0
mat_vec_prod=iterations
mat_mat_prod=2*iterations
@printf "\n\nBFGS \n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2] 
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])


@time res=newtoncg(fnc,deriv,u->approx_hessian(deriv,u),u_0)
u_2=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+iterations*400*2 
hess_eval=0
mat_vec_prod=sum(his[:,4])
mat_mat_prod=0
@printf "\n\nNewtonCG with approximated Hessian\n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2]
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])
@printf "The maximum number of cg steps was %d\n\n" maximum(his[:,4])


@time res=newtoncg(fnc,deriv,u->(p->(deriv(u+10.0^(-6)*p)-deriv(u))/10.0^(-6)),u_0)
u_3=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+sum(his[:,4])*2  #second summand: approximate action hessian: 2 evaluations per cg step
hess_eval=0
mat_vec_prod=0
mat_mat_prod=0
@printf "\n\nNewtonCG with approximated Action of Hessian\n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2]
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])
@printf "The maximum number of cg steps was %d\n\n" maximum(his[:,4])

#second starting point
u_0=zeros(N)
for i = 1:N
  u_0[i]=5+300*sin(20*pi*h*i)
end


@time res=bfgs(fnc,deriv,u_0,maxIter=100)
u_4=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+1
hess_eval=0
mat_vec_prod=iterations
mat_mat_prod=2*iterations
@printf "\n\nBFGS \n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2] 
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])

@time res=newtoncg(fnc,deriv,u->approx_hessian(deriv,u),u_0)
u_5=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+iterations*400*2  #second summand: approximate hessian: each iteration approximate hessian for two vectors (1,0) and (0,1). Each iterations requires two function evaluations
hess_eval=0
mat_vec_prod=sum(his[:,4])
mat_mat_prod=0
@printf "\n\nNewtonCG with approximated Hessian\n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2]
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])
@printf "The maximum number of cg steps was %d\n\n" maximum(his[:,4])

@time res=newtoncg(fnc,deriv,u->(p->(deriv(u+10.0^(-6)*p)-deriv(u))/10.0^(-6)),u_0)
u_6=res[1]
his=res[3]
iterations=size(his)[1]
fnc_eval=sum(his[:,3])+iterations
grad_eval=iterations+sum(his[:,4])*2  #second summand: approximate action hessian: 2 evaluations per cg step
hess_eval=0
mat_vec_prod=0
mat_mat_prod=0
@printf "\n\nNewtonCG with approximated Action of Hessian\n"
@printf "iter=%4d\t|f|=%1.2e\t|df|=%1.2e\n" iterations his[iterations,1] his[iterations,2]
@printf "fnc_eval=%d\ngrad_eval=%d\nhess_eval=%d\n" fnc_eval grad_eval hess_eval
@printf "mat_vec_prod=%d\nmat_mat_prod=%d\n" mat_vec_prod mat_mat_prod
@printf "The maximum number of line searches was %d\n\n" maximum(his[:,3])
@printf "The maximum number of cg steps was %d\n\n" maximum(his[:,4])


#Check if solutions are the same
@printf "\nAre all solutions the same?\n"
@printf "Norm(u_1-u_2)=%1.2e\n" norm(u_1-u_2)
@printf "Norm(u_1-u_3)=%1.2e\n" norm(u_1-u_3)
@printf "Norm(u_1-u_4)=%1.2e\n" norm(u_1-u_4)
@printf "Norm(u_1-u_5)=%1.2e\n" norm(u_1-u_5)
@printf "Norm(u_1-u_6)=%1.2e\n" norm(u_1-u_6)


#Plot final control, state and adjoint
domain=map(j->j*h,[1:N])
y=get_y(u_1)
p=get_p(u_1,y)
figure()
subplot(221)
plot(domain,u_1)
xlabel("t",fontdict=font1)
ylabel("u",fontdict=font1)
title("Graph of Final Control",fontdict=font1)
subplot(222)
plot(domain,y)
xlabel("t",fontdict=font1)
ylabel("y",fontdict=font1)
title("Graph of Final State",fontdict=font1)
subplot(223)
plot(domain,p)
xlabel("t",fontdict=font1)
ylabel("p",fontdict=font1)
title("Graph of Final Adjoint",fontdict=font1)


