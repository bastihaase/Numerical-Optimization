using MAT
using PyPlot
using ODE
using GridInterpolations
using OptimTools
using KrylovMethods

include("check_derivative.jl")
include("mesh.jl")
include("armijo.jl")

# For plots

font1 = Dict("family"=>"serif",
    "color"=>"black",
    "weight"=>"normal",
    "size"=>20)
vars = matread("hw4data.mat")
data = vars["data"]
grid = [(i-1)*10.0/99.0 for i=1:100]

# add noise to data

function addNoise(data,eps)
	n=size(data)[1]
	x=randn(n)
	x=eps*x
	return data+x
end

# given c and k, try to determine u

F(t,y,c,k)=[-c*y[1]-k*y[2],y[1]]

function get_u(c,k,init)
	return ode23s((t,y)->F(t,y,c,k),[0,init],grid,points=:specified)
end

# plot u and the derivative given the output of the function get_u

function plotu(inp,c,k)
	plot(inp[1],map(i->inp[2][i][2],1:size(inp[1])[1]))
	plot(inp[1],map(i->inp[2][i][1],1:size(inp[1])[1]))
	title("Plot of u and u' with c=$c and k=$k",fontdict=font1)
	xlabel("Time t in sec",fontdict=font1)
	ylabel("Displacement",fontdict=font1)
	legend(["u","u'"])
end

# determine the derivative of u(x;t) with respect to c

function get_uc(du,c,k)
	if size(du)[1]<100
		return 0
	end
	return ode23s((t,y)->[-c*y[1]-k*y[2]-interpolate(RectangleGrid(grid),du,[t]),y[1]],[0.,0.],grid,points=:specified)
end

# determine the derivative of u(x;t) with respect to k

function get_uk(u,c,k)
	if size(u)[1]<100
		return 0
	end
	return ode23s((t,y)->[-c*y[1]-k*y[2]-interpolate(RectangleGrid(grid),u,[t]),y[1]],[0.,0.],grid,points=:specified)
end

#define the function u(c,k) and its jacobian

function get_fnc_u(c,k,data)
	inp=get_u(c,k,data)
	u=map(i->inp[2][i][2],1:size(inp[1])[1])
	return u
end


function get_functions(c,k,data)
	inp=get_u(c,k,data)
	u=map(i->inp[2][i][2],1:size(inp[1])[1])
	du=map(i->inp[2][i][1],1:size(inp[1])[1])
	inp_c=get_uc(du,c,k)
	if inp_c==0
		return [0],[0],[0]
	end
	dudc=map(i->inp_c[2][i][2],1:size(inp_c[1])[1])
	inp_k=get_uk(u,c,k)
	if inp_c==0
		return [0],[0],[0]
	end
	dudk=map(i->inp_k[2][i][2],1:size(inp_k[1])[1])
	return u,dudc,dudk
end

#define objective function, Jacobian and approximated Hessian

function f(x)
	u=get_fnc_u(x[1],x[2],data[1])
	r=zeros(100)
	if size(u)[1]<100
		return -1
	end
	for i = 1:100
		r[i]=u[i]-data[i]
	end
	return 0.5*r'*r
end

function r(x)
	u=get_fnc_u(x[1],x[2],data[1])
	r=zeros(100)
	for i = 1:100
		r[i]=u[i]-data[i]
	end
	return r
end

function J(x)
	u,uc,uk=get_functions(x[1],x[2],data[1])
	J=zeros(100,2)
	for i = 1:100	
		J[i,:]=[uc[i],uk[i]]
	end
	return J
end

function df(x)
	u,uc,uk=get_functions(x[1],x[2],data[1])
	if size(u)[1]+size(uc)[1]+size(uk)[1]<300
		return [0,0]
	end
	J=zeros(100,2)
	r=zeros(100)
	for i = 1:100
		r[i]=u[i]-data[i]	
		J[i,:]=[uc[i],uk[i]]
	end
	return J'*r
end

function H(x)
	u,uc,uk=get_functions(x[1],x[2],data[1])
	J=zeros(100,2)
	if size(u)[1]+size(uc)[1]+size(uk)[1]<300
		return zeros(2,2)
	end
	for i = 1:100
		J[i,:]=[uc[i],uk[i]]
	end
	return J'*J
end

#Same thing with noise


function nf(x,init)
	u=get_fnc_u(x[1],x[2],init)
	
	r=zeros(100)
	if size(u)[1]<100
		return -1
	end
	
	for i = 1:100
		r[i]=u[i]-data[i]
	end
	return 0.5*r'*r
end

function ndf(x,init)
	u,uc,uk=get_functions(x[1],x[2],init)
	J=zeros(100,2)
	r=zeros(100)
	if size(u)[1]+size(uc)[1]+size(uk)[1]<300
		return [0,0]
	end
	for i = 1:100
		r[i]=u[i]-data[i]
		J[i,:]=[uc[i],uk[i]]
	end
	return J'*r
end

function nH(x,init)
	u,uc,uk=get_functions(x[1],x[2],init)
	J=zeros(100,2)
	if size(u)[1]+size(uc)[1]+size(uk)[1]<300
		return zeros(2,2)
	end
	for i = 1:100
		J[i,:]=[uc[i],uk[i]]
	end
	return J'*J
end













	
