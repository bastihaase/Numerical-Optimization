using PyPlot


#technical function to resolve type conversion issues,i.e. 1xnxm is not identified with nxm arrays/matrices

function triple_double(M)
  n=size(M)
  res=zeros(n[2],n[3])
  for i = 1:n[2]
    for j = 1:n[3]
      res[i,j]=M[1,i,j]
    end
  end
  return res
end

#functions to apply fnc and gradient to the test vectors

function apply_fnc(f,inp,dim)
      result=zeros(10,dim)
      for i = 1:10
        result[i,:]=f(inp[i,:]')
      end
      return result
end

function det_mat(deriv,inp,dim)
  result=zeros(10,dim[1],dim[2])
  for i=1:10
    result[i,:,:]=deriv(inp[i,:]')
  end
  return result
end

function apply_grad(m,v,dim)
  res=zeros(10,dim)
  for i = 1:10
    res[i,:]=(triple_double(m[i,:,:]))'*v[i,:]'
  end
  return res
end

function apply_hessian(hess,v,dim)
  res=zeros(10,1)
  for i= 1:10
    res[i,:]=v[i,:]*triple_double(hess[i,:,:])*v[i,:]'
  end
  return res
end


#actual check derivative

function check_derivative(f,deriv,dim_inp,dim_f)
  x0 = randn(10,dim_inp)
  v  = randn(size(x0))
  f0 = apply_fnc(f,x0,dim_f)
  df0 = det_mat(deriv,x0,[dim_inp,dim_f])
  Error = zeros(10,2)
  w=apply_grad(df0,v,dim_f)
  for j=1:10
    ft = apply_fnc(f,(x0+10.0^(-j)*v),dim_f)
    Error[j,1] = norm(f0-ft)/norm(f0)
    Error[j,2] = norm(f0 .+10.0^(-j)*w .- ft)/norm(f0)
    #println(@sprintf("%1.3e\t%1.3e\t%1.3e",10.0^(-j), Error[j,1],Error[j,2]))
  end
  loglog(logspace(-1,-10,10),Error)
  legend(("|f(x) - T0(x)|","|f(x) - T1(x)|"))
  xlabel("Distance")
  ylabel("Error")
end

#check derivative at a specified point

function check_derivative_at_point(f,deriv,dim_inp,dim_f,x0)
  v  = randn(size(x0))
  f0=f(x0)
  df0=deriv(x0)
  Error = zeros(10,2)
  for j=1:10
    ft = f(x0+10.0^(-j)*v)
    Error[j,1] = norm(f0-ft)/norm(f0)
    Error[j,2] = norm(f0 .+10.0^(-j)*df0'*v .- ft)/norm(f0)
    #println(@sprintf("%1.3e\t%1.3e\t%1.3e",10.0^(-j), Error[j,1],Error[j,2]))
  end
  loglog(logspace(-1,-10,10),Error)
  legend(("|f(x) - T0(x)|","|f(x) - T1(x)|"))
  xlabel("Distance")
  ylabel("Error")
end

function check_hessian(f,deriv,hessian,dim_inp) # currently f needs to be a a scalar valued function
  x0 = randn(10,dim_inp)
  v  = randn(size(x0))
  f0 = apply_fnc(f,x0,1)
  df0 = det_mat(deriv,x0,[dim_inp,1])
  d2f0 = det_mat(hessian,x0,[dim_inp,dim_inp])
  Error = zeros(10,2)
  w=apply_grad(df0,v,1)
  w2=apply_hessian(d2f0,v,dim_inp)
  for j=1:10
    ft = apply_fnc(f,(x0+10.0^(-j)*v),1)
    Error[j,1] = norm(f0-ft)/norm(f0)
    Error[j,2] = norm(f0 .+10.0^(-j)*w .+0.5*10.0^(-2*j)*w2  .- ft)/norm(f0)
    #println(@sprintf("%1.3e\t%1.3e\t%1.3e",10.0^(-j), Error[j,1],Error[j,2]))
  end
  loglog(logspace(-1,-10,10),Error)
  legend(("|f(x) - T0(x)|","|f(x) - T2(x)|"))
xlabel("Distance")
  ylabel("Error")
end





