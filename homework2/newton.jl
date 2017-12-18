using KrylovMethods

function failure_cholesky(A,beta=10.0^(-3))
  mat_fac=0
  k=size(A)[1]
  a=minimum(A)
  t=0
  w=A
  if a <= 0
    t=-a+beta
  end
  success = 0
  while true
    try w=cholfact(A+t*eye(k))
    catch PosDefException
      t=max(2*t,beta)
      success = 1
    end
    mat_fac+=1
    if success==0
      break
    end
  end
  return w,[0,0,0,mat_fac,0]
end


function spectrum(A,beta=10.0^(-3))
  k=size(A)[1]
  F=schurfact(A)
  a=minimum(F[:values])
  if a>0
    return A
  end
  return A+(-a+beta)*eye(k)
end

#different methods implememnted

function apply_steep(f::Function,fc,df,xc,maxIter,c1,b)
  fnc_eval=0
  LS = 1
  t  = 1
  pc=(-1)*df
  while LS<=maxIter
    fnc_eval+=1
    if f(xc+t*pc)[1,1] <= (fc + t*c1*dot(vec(df),vec(pc)))
        break
    end
    t *= b
    LS += 1
  end
  if LS>maxIter
	   LS= -1
	   t = 0.0
  end
  return xc+t*pc,LS,[fnc_eval,0,0,0,0]
end

function apply_fcol(f::Function,fc,df,d2f,xc,maxIter,c1,b)
  chol=d2f,[0,0,0,1,0]
  try cholfact(d2f)
  catch y
    chol=failure_cholesky(d2f)
    chol[2][4]+=1
  end
  pc=(-1)*\(chol[1],df)
  chol[2][5]+=1
  LS = 1
  t  = 1
  while LS<=maxIter
    chol[2][1]+=1
    if f(xc+t*pc)[1,1] <= (fc + t*c1*dot(vec(df),vec(pc)))
        break
    end
    t *= b
    LS += 1
  end
  if LS>maxIter
    LS= -1
    t = 0.0
  end
  return xc+t*pc,LS,chol[2]
end

function apply_spec(f::Function,fc,df,d2f,xc,maxIter,c1,b)
  spec=spectrum(d2f)
  pc=(-1)*\(cholfact(spec),df)
  eval=[0,0,0,2,1]
  LS = 1
  t  = 1
  while LS<=maxIter
    eval[1]+=1
    if f(xc+t*pc)[1,1] <= (fc + t*c1*dot(vec(df),vec(pc)))
        break
    end
    t *= b
    LS += 1
  end
  if LS>maxIter
	   LS= -1
	   t = 0.0
  end
  return xc+t*pc,LS,eval
end

function apply_cg(f::Function,fc,df,d2f,xc,maxIter,c1,b)
  res=cg(d2f,df,maxIter=1000,out=-1)
  pc=(-1)*res[1]
  mat_mult=res[4]
  fnc_eval=0
  LS = 1
  t  = 1
  while LS<=maxIter
    fnc_eval+=1
    if f(xc+t*pc)[1,1] <= (fc + t*c1*dot(vec(df),vec(pc)))
        break
    end
    t *= b
    LS += 1
  end
  if LS>maxIter
	   LS= -1
	   t = 0.0
  end
  return xc+t*pc,LS,[fnc_eval,0,0,0,mat_mult]
end

#method choose method: steep=steppest descent, fcol= failure of cholesky, spec=analyze spectru, cg=conjugent gradient

function armijo(f::Function,fc,df,d2f,xc,method="steep";maxIter=100, c1=1e-4,b=0.5)
  if method=="steep"
    return apply_steep(f,fc,df,xc,maxIter,c1,b)
  end
  if method=="fcol"
    return apply_fcol(f,fc,df,d2f,xc,maxIter,c1,b)
  end
  if method=="spec"
    return apply_spec(f,fc,df,d2f,xc,maxIter,c1,b)
  end
  if method=="cg"
    return apply_cg(f,fc,df,d2f,xc,maxIter,c1,b)
  end
end

function opti(f::Function,J::Function,H::Function,x::Vector,method::String;maxIter=100000,atol=1e-6,doPrint=false)
    w=[0,0,0,0,0] #[fnc_eval,df_eval,hess_eval,mat_fact,mat_mult]
    his = zeros(maxIter,2)
    n = length(x)
    I = eye(n)
    i = 1
    xOld = x
    df   = J(x)  #count these as function evaluations?
    d2f = H(x)
    w[1]+=1
    w[2]+=1
    while i<=maxIter
        fc = f(x)[1,1]
        w[1]=w[1]+1
        his[i,:] = [norm(fc) norm(df)]
        # line search
        xk,LS,eval = armijo(f,fc,df,d2f,x,method)
        if doPrint
            @printf "iter=%04d\t|f|=%1.2e\t|df|=%1.2e\tLS=%d\n" i his[i,1] his[i,2] LS
        end
		    if LS==-1
			      @printf "Linesearch failed!"
            his = his[1:i,:]
			      break;
		    end

        # update x and H
        w= w + eval
        x   = xk
        df  = J(x)
        d2f = H(x)
        w[1]+=1
        w[2]+=1
        w[3]+=1
        if(norm(df)<atol)
            his = his[1:i,:]
            break
        end
        i+=1
    end
    return x,his,w,i
end


