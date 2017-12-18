function Classical(x)
	res=0
	for i=1:10
		res+=(exp(-x[1]*i*0.1)-exp(-x[2]*i*0.1)-x[3]*(exp(-0.1*i)-exp(-i)))^2
	end
	return 0.5*res
end

function grad_Classical(x)
	J_Classical(x)'*r_Classical(x)
end

function JTJ_Classical(x)
	J=J_Classical(x)
	return J'*J
end

function r_Classical(x)
	res=zeros(10)
	for i=1:10
		res[i]=exp(-x[1]*i*0.1)-exp(-x[2]*i*0.1)-x[3]*(exp(-0.1*i)-exp(-i))
	end
	return res
end

function J_Classical(x)
	J=zeros(10,3)
	for i=1:10
		J[i,1]=-0.1*i*exp(-x[1]*i*0.1)
		J[i,2]=0.1*i*exp(-x[2]*i*0.1)
		J[i,3]=-exp(-0.1*i)+exp(-i)
	end
	return J
end
	
function S_Classical(x)
	hess=zeros(3,3,10)
	for i=1:10
		hess[:,1,i]=[i^2/100*exp(-i/10*x[1]),0,0]
		hess[:,2,i]=[0,-i^2/100*exp(-i/10*x[2]),0]
	end
	res=zeros(3,3)
	rx=r_Classical(x)
	for i=1:10
		res+=rx[i]*hess[:,:,i]
	end
	return res
end

function Hess_Classical(x)
	return JTJ_Classical(x)+S_Classical(x)
end	
		
