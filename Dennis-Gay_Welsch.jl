function Dennis_Gay_Welsch(f,grad,J,r,x_0,n;maxIter=100,atol=10.0^(-8))
	B=zeros(n,n)
	xc=x_0
	iter=0
	rc=r(xc)
	Jc=J(xc)
	performance=[0,1,1,1,0]#eval f, eval grad J, matrix matrix, matrix vector,matrix fact
	df=Jc'*rc
	while(norm(df)>atol && iter<=maxIter)
		H=Jc'*Jc+B
		pk = \(H,-df) 
		xc_new=xc+pk
		rc_new=r(xc_new)
		J_new=J(xc_new)
		df=J_new'*rc_new
		#update everything
		performance+=[0,1,2,1,1]
		s=xc_new-xc
		y=J_new'*rc_new-Jc'*rc
		y2=J_new'*rc_new-Jc'*rc_new
		performance+=[0,0,2,0,0]
		#scale B
		tau=abs((s'*y2/(s'*B*s))[1])
		B=min(1,tau)*B
		#update B
		B=B+((y2-B*s)*y'+y*(y2-B*s)')/(y'*s)[1]-((y2-B*s)'*s/((y'*s)[1])^2)[1]*y*y'
		performance+=[0,0,4,7,0]
		iter=iter+1
		Jc=J_new
		xc=xc_new
		rc=rc_new
	end
	return xc,iter,norm(df),performance
end
