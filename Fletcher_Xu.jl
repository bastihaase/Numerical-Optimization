function Fletcher_Xu(f,grad,J,r,x_0;maxIter=100,atol=10.0^(-8),tau=0.2)
	fc=f(x_0)
	Jc=J(x_0)
	B=Jc'*Jc
	rc=r(x_0)
	dc=Jc'*rc
	xc=x_0
	n=size(x_0)[1]
	performance=[1,1,2,0,0] #eval f, eval grad J, matrix matrix, matrix vector,matrix fact 
	iter=0
	Gauss=0
	bfgs=0
	while(norm(dc)>atol && iter<=maxIter)
		
		pk=\(B,-dc)
		ak=my_armijo(f,grad,fc,dc,xc,pk)
		xc_new=xc+ak[1]*pk
		
		fc_new=f(xc_new)
		Jc_new=J(xc_new)
		rc=r(xc_new)
		performance+=[1+ak[2],2,0,0,1]
		if (fc-fc_new)[1]/fc[1]>= tau*fc[1]
			#Gauss-Newton
			Gauss=Gauss+1
			B=Jc_new'*Jc_new
			performance+=[0,0,1,0,0]
		else
			#modified BFGS
			bfgs=bfgs+1
			delta_x=xc_new-xc
			gamma_new=Jc_new'*Jc_new*delta_x+(Jc_new-Jc)'*rc
			gamma=grad(xc_new)-grad(xc)
			if (delta_x'*gamma_new)[1] >= (0.01* delta_x'*gamma)[1]
				gamma=gamma_new
			end
			B=B+gamma*gamma'/(gamma'*delta_x)[1]-(B*delta_x*delta_x'*B)/(delta_x'*B*delta_x)[1]
			performance+=[0,0,3,5,0]
		end
		xc=xc_new
		fc=fc_new
		Jc=Jc_new
		dc=grad(xc)
		iter=iter+1
	end
	
	return xc,iter+1,norm(grad(xc)),[Gauss+1,bfgs],performance
end

function my_Gauss_Newton(f,grad,J,r,x_0;maxIter=100,atol=10.0^(-8))
	fc=f(x_0)
	Jc=J(x_0)
	B=Jc'*Jc
	rc=r(x_0)
	dc=Jc'*rc
	xc=x_0
	performance=[1,1,2,0,0] #eval f, eval grad J, matrix matrix, matrix vector,matrix fact 
	iter=0
	while(norm(Jc'*rc)>atol && iter<=maxIter)
		pk=\(B,-dc)
		ak=my_armijo(f,grad,fc,dc,xc,pk)
		xc_new=xc+ak[1]*pk
		
		fc_new=f(xc_new)
		Jc_new=J(xc_new)
		rc=r(xc_new)
			
		B=Jc_new'*Jc_new
		performance+=[1+ak[2],2,1,0,1]
		xc=xc_new
		fc=fc_new
		Jc=Jc_new
		dc=grad(xc)
		iter=iter+1
	end
	return xc,iter,norm(grad(xc)),performance
end

function my_bfgs(f,grad,J,r,x_0;maxIter=100,atol=10.0^(-8),tau=0.2)
	fc=f(x_0)
	Jc=J(x_0)
	n=size(x_0)[1]
	B=Jc'*Jc
	rc=r(x_0)
	dc=Jc'*rc
	xc=x_0
	
	performance=[1,1,2,0,0] #eval f, eval grad J, matrix matrix, matrix vector,matrix fact 

	iter=0
	
	while(norm(Jc'*rc)>atol && iter<=maxIter)
    		
		pk=\(B,-dc)
		ak=my_armijo(f,grad,fc,dc,xc,pk)
		xc_new=xc+ak[1]*pk
		
		fc_new=f(xc_new)
		Jc_new=J(xc_new)
		rc=r(xc_new)
		
			
		delta_x=xc_new-xc
		gamma_new=Jc_new'*Jc_new*delta_x+(Jc_new-Jc)'*rc
		gamma=grad(xc_new)-grad(xc)
		if (delta_x'*gamma_new)[1] >= (0.01* delta_x'*gamma)[1]
			gamma=gamma_new
		end
		B=B+gamma*gamma'/(gamma'*delta_x)[1]-(B*delta_x*delta_x'*B)/(delta_x'*B*delta_x)[1]
		performance+=[1+ak[2],2,3,5,1]
		xc=xc_new
		fc=fc_new
		Jc=Jc_new
		dc=grad(xc)
		iter=iter+1
	end
	return xc,iter,norm(grad(xc)),performance
end




