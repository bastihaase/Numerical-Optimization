using PyPlot
include("check_derivative.jl")
include("newton.jl")

#function to plot contour plots easily

function create_contour_data(fnc,steps=10,start=-1,stop=1)
  res=zeros(steps,steps)
  x=linspace(start,stop,steps)
  y=linspace(start,stop,steps)
  for i = 1:steps
    for j = 1:steps
      res[i,j]=fnc(x[i],y[j])
    end
  end
  return res
end


#diagm can not deal with 1xn matrices

function convert_matrix_vector(A)
  if size(A)[1]==1
    n=size(A)[2]
    ret=zeros(n)
    for i = 1:n
      ret[i]=A[1,i]
    end
    return ret
  end
  n=size(A)[1]
  ret=zeros(n)
  for i= 1:n
      ret[i]=A[i,1]
  end
  return ret
end

# various functions to initialize the required matrices for arbitrary parameters


function get_A(n)
  A=eye(n)-diagm(ones(n-1),-1)
  A=A[:,1:(n-1)]
  return [A zeros(n,n-1);  zeros(n,n-1) A]
end

function get_M(n)
  M=eye(n)+diagm(ones(n-1),-1)
  M=M[:,1:(n-1)]
  return [M;  M]
end

function get_v(n,a,b)
  v=zeros(2*n,1)
  v[1]=-a[1]
  v[n]=b[1]
  v[n+1]=-a[2]
  v[2*n]=b[2]
  return v
end

function get_w(n,rho,a,b)
  w=zeros(2*n,1)
  w[1]=rho(a)
  w[n+1]=rho(a)
  w[n]=rho(b)
  w[2*n]=rho(b)
  return w
end

function get_rho_x(n,x,rho)
  res=zeros(n-1,1)
  for i= 1:n-1
    res[i]=rho([x[i],x[i+n-1]])
  end
  return res
end


#set alpha, beta and define rho and its jacobian

alpha=2
beta=5

r(x)=1+alpha*exp(-beta*(x[1]^2+x[2]^2))


function get_jacob_rho(n,x)
  m=zeros(2*(n-1),n-1)
  for i = 1:(n-1)
    m[i,i]=-2*alpha*beta*x[i]*exp(-beta*(x[i]^2+x[n-1+i]^2))
    m[n-1+i,i]=-2*alpha*beta*x[i+n-1]*exp(-beta*(x[i]^2+x[n-1+i]^2))
  end
  return m
end

function exp_2(x)
  return exp(-beta*(x[1]^2+x[2]^2))
end


# Set parameters and define objective function and its jacobian

n=10
a=[1 1]
b=[-1 -1]

A=get_A(n)
v=get_v(n,a,b)
w=get_w(n,r,a,b)
M=get_M(n)
M'
g(x)=(1/(2*n)*(((A*x+v).^2)')*(M*get_rho_x(n,x,r)+w))[1]
Jacob_g(x)=vec(1/(2*n)*(get_jacob_rho(n,x)*M'*((A*x+v)).^2+2*(A'*diagm(vec(A*x+v)))*(M*get_rho_x(n,x,r)+w)))

#build hessian


function part1(x)
  r=M'*(A*x+v).^2
  res=zeros(2*(n-1),2*(n-1))
  for i = 1:(n-1)
    res[i,i]=-2*alpha*beta*(1-2*beta*x[i]^2)*exp_2([x[i],x[n+i-1]])*r[i]
    res[n-1+i,i]=4*alpha*beta^2*x[i]*x[n-1+i]*exp_2([x[i],x[n+i-1]])*r[i]
  end
  for i = n:(2*(n-1))
    res[i,i]=-2*alpha*beta*(1-2*beta*x[i]^2)*exp_2([x[-n+1+i],x[i]])*r[-n+1+i]
    res[-n+1+i,i]=4*alpha*beta^2*x[i]*x[-n+1+i]*exp_2([x[i],x[-n+1+i]])*r[-n+1+i]
  end
  return res
end

function part2(x)
  return 2*A'*diagm(vec(A*x+v))*M*get_jacob_rho(n,x)'
end

function part3(x)
  return 2*(A'*diagm(vec(M*get_rho_x(n,x,r)+w))+get_jacob_rho(n,x)*M'*diagm(vec(A*x+v)))*A
end

#check individual parts

t(x)=get_jacob_rho(n,x)*M'*(A*x+v).^2
h(x)=part1(x)+part2(x)
p(x)=part3(x)
j(x)=2*A'*diagm(vec((A*x+v)))*(M*get_rho_x(n,x,r)+w)
check_derivative(j,p,2*(n-1),2*(n-1))

#check the whole hessian

hessian_g(x)=1/(2*n)*(part1(x)+part2(x)+part3(x))
check_derivative(g,Jacob_g,2*(n-1),1)
check_hessian(g,Jacob_g,hessian_g,2*(n-1))

#Apply all methods
#First point
start=[0.8,0.8,0.8,0.4,0,-0.2,-0.4,-0.6,-0.8,0.8,0.6,0.4,0.2,0,-0.2,-0.4,-0.6,-0.8]
eigvals(hessian_g(start))
#not positive definite
res=opti(g,Jacob_g,hessian_g,start,"steep")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
contour(linspace(-1,1,10),linspace(-1,1,10),create_contour_data((x,y)->1+exp(-1(x^2+y^2))))
legend(("Start Value","End Value"))
title("Contourplot with steep")
xlabel("x")
ylabel("y")
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient steep")
xlabel("Iterations")
ylabel("Norm of Gradient")

res=opti(g,Jacob_g,hessian_g,start,"fcol")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
contour(linspace(-1,1,10),linspace(-1,1,10),create_contour_data((x,y)->1+exp(-1(x^2+y^2))))
legend(("Start Value","End Value"))
title("Contourplot with fcol")
xlabel("x")
ylabel("y")
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient fcol")
xlabel("Iterations")
ylabel("Norm of Gradient")

res=opti(g,Jacob_g,hessian_g,start,"spec")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
contour(linspace(-1,1,10),linspace(-1,1,10),create_contour_data((x,y)->1+exp(-1(x^2+y^2))))
legend(("Start Value","End Value"))
title("Contourplot with spec")
xlabel("x")
ylabel("y")
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient spec")
xlabel("Iterations")
ylabel("Norm of Gradient")

res=opti(g,Jacob_g,hessian_g,start,"cg")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
contour(linspace(-1,1,10),linspace(-1,1,10),create_contour_data((x,y)->1+exp(-1(x^2+y^2))))
legend(("Start Value","End Value"))
title("Contourplot with cg")
xlabel("x")
ylabel("y")
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient cg")
xlabel("Iterations")
ylabel("Norm of Gradient")

#Second point # just for me: data not submitted
start=[0.8,0.8,0.8,0.8,0.8,0.4,0,-0.4,-0.8,0.8,0.4,0,-0.4,-0.8,-0.8,-0.8,-0.8,-0.8]
eigvals(hessian_g(start))
# positive definite
res=opti(g,Jacob_g,hessian_g,start,"steep")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")

res=opti(g,Jacob_g,hessian_g,start,"fcol")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")

res=opti(g,Jacob_g,hessian_g,start,"spec")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")

res=opti(g,Jacob_g,hessian_g,start,"cg")
data=res[1]
res[2]
res[3]
res[4]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
