function Nielsen(x)
	res=0
	y=[2,0,2/3,0,2/5,0,2/7,0,2/9,0]	
	for i=0:9
		res+=(x[1]*x[3]^i+x[2]*x[4]^i-y[i+1])^2
	end
	return 0.5*res
end

function grad_Nielsen(x)
	J_Nielsen(x)'*r_Nielsen(x)
end

function JTJ_Nielsen(x)
	J=J_Nielsen(x)
	return J'*J
end

function r_Nielsen(x)
	res=zeros(10)
	y=[2,0,2/3,0,2/5,0,2/7,0,2/9,0]	
	for i=0:9
		res[i+1]=x[1]*x[3]^i+x[2]*x[4]^i-y[i+1]
	end
	return res
end

function J_Nielsen(x)
	J=zeros(4,10)
	for i=0:9
		J[1,i+1]=x[3]^i
		J[2,i+1]=x[4]^i
		J[3,i+1]=i*x[1]*x[3]^(i-1)
		J[4,i+1]=i*x[2]*x[4]^(i-1)
	end
	return J'
end

function S_Nielsen(x)
	hess=zeros(4,4,10)
	for i=0:9
		hess[:,1,i+1]=[0,0,i*x[3]^(i-1),0]
		hess[:,2,i+1]=[0,0,0,i*x[4]^(i-1)]
		hess[:,3,i+1]=[i*x[3]^(i-1),0,i*(i-1)*x[1]*x[3]^(i-2),0]
		hess[:,4,i+1]=[0,i*x[4]^(i-1),0,i*(i-1)*x[2]*x[4]^(i-2)]
	end
	res=zeros(4,4)
	rx=r_Nielsen(x)
	for i=1:10
		res+=rx[i]*hess[:,:,i]
	end
	return res
end

function Hess_Nielsen(x)
	return JTJ_Nielsen(x)+S_Nielsen(x)
end	
	
		
