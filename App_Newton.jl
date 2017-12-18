function App_Newton(f,grad,J,r,x_0,n,m;maxIter=100,atol=10.0^(-8))
	B=zeros(n,n,m)
	for i=1:m
		B[:,:,i]=approx_hessian(x->J(x)[i,:],x_0)
	end
	xc=x_0
	H=zeros(n,n)
	iter=0
	rc=r(xc)
	Jc=J(xc)
	df=Jc'*rc
	performance=[0,m*2+1,0,0,0] #eval f, eval grad J, matrix matrix, matrix vector,matrix fact
	while(norm(df)>atol && iter<=maxIter)
		H=Jc'*Jc
		for j=1:m
			H+=rc[j]*B[:,:,j]
		end
		pk = \(H,-df) 
		xc_new=xc+pk
		rc=r(xc_new)
		J_new=J(xc_new)
		df=J_new'*rc
		#update everything
		for i=1:m
			B[:,:,i]=approx_hessian(x->J(x)[i,:],xc_new)
		end
		performance+=[0,3,0,1,1]
		iter=iter+1
		Jc=J_new
		xc=xc_new
	end
	return xc,iter,norm(df),performance
end


function approx_action_of_hessian(grad,x,p,eps=10.0^(-6))
  return (grad(x+eps*p)-grad(x))/eps
end

function approx_hessian(grad,x,eps=10.0^(-6))
  n=size(x)[1]
  hess=zeros(n,n)
  for i = 1:n
    p=zeros(n)
    p[i]=1
    hess[:,i]=approx_action_of_hessian(grad,x,p,eps)
  end
  return hess
end
