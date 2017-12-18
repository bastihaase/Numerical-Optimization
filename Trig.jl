function Trig(x)
	res=0
	for i=1:20
		res+=((x[1]+x[2]*0.2*i-exp(0.2*i))^2+(x[3]+x[4]*sin(0.2*i)-cos(0.2*i))^2)^2
	end
	return 0.5*res
end

function grad_Trig(x)
	J_Trig(x)'*r_Trig(x)
end

function JTJ_Trig(x)
	J=J_Trig(x)
	return J'*J
end

function r_Trig(x)
	res=zeros(20)
	for i=1:20
		res[i]=(x[1]+x[2]*0.2*i-exp(0.2*i))^2+(x[3]+x[4]*sin(0.2*i)-cos(0.2*i))^2
	end
	return res
end

function J_Trig(x)
	J=zeros(20,4)
	for i=1:20
		J[i,1]=2*(x[1]+x[2]*0.2*i-exp(0.2*i))
		J[i,2]=0.4*(x[1]+x[2]*0.2*i-exp(0.2*i))*i
		J[i,3]=2*(x[3]+x[4]*sin(0.2*i)-cos(0.2*i))
		J[i,4]=2*(x[3]+x[4]*sin(0.2*i)-cos(0.2*i))*sin(0.2*i)
	end
	return J
end
	
function S_Trig(x)
	hess=zeros(4,4,20)
	for i=1:20
		hess[:,1,i]=[2.0,0.4*i,0,0]
		hess[:,2,i]=[0.4*i,0.08*i^2,0,0]
		hess[:,3,i]=[0,0,2.0,2.0*sin(0.2*i)]
		hess[:,4,i]=[0,0,2.0*sin(0.2*i),2.0*(sin(0.2*i))^2]
	end
	res=zeros(4,4)
	rx=r_Trig(x)
	for i=1:20
		res+=rx[i]*hess[:,:,i]
	end
	return res
end

function Hess_Trig(x)
	return JTJ_Trig(x)+S_Trig(x)
end		
		
