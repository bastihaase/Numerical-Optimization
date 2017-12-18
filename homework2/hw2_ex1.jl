include("check_derivative.jl")
include("newton.jl")

#function to create nxn mesh on the interval given by x and y

function create_mesh(n,x,y)
  res = zeros(n,n,2)
  lx = (x[2]-x[1])/n
  ly = (y[2]-y[1])/n
  for i = 1:n
    for j = 1:n
      res[i,j,:]=[x[1]+(j-0.5)*lx,y[2]-(i-0.5)*ly]
    end
  end
  return res
end

#Rosenbrock functions and first two derivatives


f(x)=(1-x[1])^2+100*(x[2]-x[1]^2)^2
grad(x)=[-2*(1-x[1])-400*(x[1]*x[2]-x[1]^3),200*(x[2]-x[1]^2)]
hess(x)=[+2-400*x[2]+1200*x[1]^2 -400*x[1] ; -400*x[1] 200]

#Check derivatives
check_hessian(f,grad,hess,2)

#Create the required mesh for the exercise
mesh=create_mesh(32,[-2 2],[-1 3])

# check that starting point (-1.9375,-0.9375) has a spd hessian for performance comparison
mesh[1,1,:]
eigvals(hess(mesh[1,1,:]))

#apply each method to this point
res=opti(f,grad,hess,vec(mesh[1,1,:]),"steep")
res[1]
res[2]
res[3] #ignore hessian evaluations at this point
res[4]
plot([1000:res[4]],res[2][1000:res[4],2])
title("Gradient steep SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")

res=opti(f,grad,hess,vec(mesh[1,1,:]),"fcol")
res[1]
res[2]
res[3]
res[4]
plot([5:25],res[2][5:25,2])
title("Gradient fcol SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")


res=opti(f,grad,hess,vec(mesh[1,1,:]),"spec")
res[1]
res[2]
res[3]
res[4]
plot([5:res[4]],res[2][5:res[4],2])
title("Gradient spec SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")


res=opti(f,grad,hess,vec(mesh[1,1,:]),"cg")
res[1]
res[2]
res[3]
res[4]
plot([5:res[4]],res[2][5:res[4],2])
title("Gradient cg SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")


#check that starting point (-0.3125,0.6875) has a nondefinite hessian for function evaluation part
mesh[14,20,:]
eigvals(hess(mesh[14,20,:]))

#apply each method to this point
res=opti(f,grad,hess,vec(mesh[14,20,:]),"steep")
res[1]
res[2]
res[3] #ignore evaluations of hessian
res[4]
plot([100:res[4]-1],res[2][100:res[4]-1,2])
title("Gradient steep Non-SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")

res=opti(f,grad,hess,vec(mesh[14,20,:]),"fcol")
res[1]
res[2]
res[3]
res[4]
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient fcol Non-SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")


res=opti(f,grad,hess,vec(mesh[14,20,:]),"spec")
res[1]
res[2]
res[3]
res[4]
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient spec Non-SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")


res=opti(f,grad,hess,vec(mesh[14,20,:]),"cg")
res[1]
res[2]
res[3]
res[4]
plot([1:res[4]],res[2][1:res[4],2])
title("Gradient cg Non-SPD")
xlabel("Iterations")
ylabel("Norm of Gradient")



# create matrix for illustration of region of convergence
region=zeros(32,32)


#apply each method to all points of the region and plot where it is successful

for i = 1:32
  for j = 1:32
    res=opti(f,grad,hess,vec(mesh[i,j,:]),"steep")
    if norm(res[1]-[1,1])<10.0^(-4)
      region[i,j]=1
    else
      region[i,j]=0
    end
  end
end
region
spy(region)
title("Region of Convergence steep")

for i = 1:32
  for j = 1:32
    res=opti(f,grad,hess,vec(mesh[i,j,:]),"fcol")
    if norm(res[1]-[1,1])<10.0^(-4)
      region[i,j]=1
    else
      region[i,j]=0
    end
  end
end
region
spy(region)
title("Region of Convergence fcol")

for i = 1:32
  for j = 1:32
    res=opti(f,grad,hess,vec(mesh[i,j,:]),"spec")
    if norm(res[1]-[1,1])<10.0^(-4)
      region[i,j]=1
    else
      region[i,j]=0
    end
  end
end
region
spy(region)
title("Region of Convergence spec")

for i = 1:32
  for j = 1:32
    res=opti(f,grad,hess,vec(mesh[i,j,:]),"cg")
    if norm(res[1]-[1,1])<10.0^(-4)
      region[i,j]=1
    else
      region[i,j]=0
    end
  end
end
region
spy(region)
title("Region of Convergence cg")

