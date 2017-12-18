using PyPlot
include("check_derivative.jl")
include("gd.jl")

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


# Set paramters and define objective function and its jacobian

n=10
a=[1 1]
b=[-1 -1]


A=get_A(n)
v=get_v(n,a,b)
w=get_w(n,r,a,b)
M=get_M(n)

g(x)=1/(2*n)*(((A*x+v).^2)')*(M*get_rho_x(n,x,r)+w)
Jacob_g(x)=1/(2*n)*(get_jacob_rho(n,x)*M'*((A*x+v)).^2+2*(A'*diagm(convert_matrix_vector(A*x+v)))*(M*get_rho_x(n,x,r)+w))


# check the derivative by means of the provided script


check_derivative(g,Jacob_g,2*(n-1),1)


#Apply steepest descent method by use of the provided script

start=[0.8,0.8,0.8,0.4,0,-0.2,-0.4,-0.6,-0.8,0.8,0.6,0.4,0.2,0,-0.2,-0.4,-0.6,-0.8]
res=gd(g,Jacob_g,start)
data=res[1]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")


start=[0.8,0.7,0.4,0.4,0,-0.2,-0.4,-0.5,-0.9,0.8,0.7,0.6,0,0,-0.2,-0.6,-0.7,-0.8]
res=gd(g,Jacob_g,start)
data=res[1]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")

start=[0.8,0.8,0.8,0.8,0.8,0.4,0,-0.4,-0.8,0.8,0.4,0,-0.4,-0.8,-0.8,-0.8,-0.8,-0.8]
res=gd(g,Jacob_g,start)
data=res[1]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")

start=randn(18)
start=start/(norm(start))
res=gd(g,Jacob_g,start)
data=res[1]
plot(start[1:9],start[10:18],marker="o")
plot(data[1:9],data[10:18],marker="o")
contour(linspace(-1,1,10),linspace(-1,1,10),create_contour_data((x,y)->1+exp(-1(x^2+y^2))))



