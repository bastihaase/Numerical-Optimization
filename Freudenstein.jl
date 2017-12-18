# Freudenstein function


function Freudenstein(x)
	return 0.5*((x[1]-13+((5-x[2])*x[2]-2)*x[2])^2+(x[1]-29+((x[2]+1)*x[2]-14)*x[2])^2)
end

function grad_Freudenstein(x)
	J_Freudenstein(x)'*r_Freudenstein(x)
end

function JTJ_Freudenstein(x)
	J=J_Freudenstein(x)
	return J'*J
end

function J_Freudenstein(x)
	return [1 1; 10*x[2]-3*x[2]^2-2 3*x[2]^2+2*x[2]-14]'
end

function r_Freudenstein(x)
	return [(x[1]-13+((5-x[2])*x[2]-2)*x[2]),(x[1]-29+((x[2]+1)*x[2]-14)*x[2])]
end

function S_Freudenstein(x)
	hess=zeros(2,2,2)
	hess[2,2,1]=10-6*x[2]
	hess[2,2,2]=6*x[2]+2
	res=zeros(2,2)
	rx=r_Freudenstein(x)
	for i=1:2
		res+=rx[i]*hess[:,:,i]
	end
	return res
end

function Hess_Freudenstein(x)
	return JTJ_Freudenstein(x)+S_Freudenstein(x)
end
